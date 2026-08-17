package store

import (
	"context"
	"log"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/Gentleman-Programming/engram/internal/embed"
)

// DefaultSimilarityFloor is the per-dense-branch guard against noise. It is
// deliberately lax: ranking is the primary signal, the floor only drops
// candidates that have nothing to do with the query (measured healthy
// similarities start around 0.30; a fixed high threshold was refuted).
// Override with ENGRAM_EMBED_FLOOR.
const DefaultSimilarityFloor = 0.15

// embedTimeout bounds a query embedding. Loading two models on a cold
// process takes seconds; this is generous enough for that and still bounded.
const embedTimeout = 30 * time.Second

func similarityFloor() float64 {
	if v := os.Getenv("ENGRAM_EMBED_FLOOR"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return DefaultSimilarityFloor
}

// rankedBranch is one ranking plus its per-id scores. Scores from different
// branches live on incomparable scales (BM25 vs cosines of two models); they
// break ties on equal rank and are never averaged across branches.
type rankedBranch struct {
	ids   []int64
	score map[int64]float64
}

// cosine assumes vectors normalized to unit length, so it is a dot product.
// Mismatched lengths (different models) return 0 instead of breaking.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

// fuseMinRank fuses rankings by best position: each id's final rank is the
// best (lowest) position it reached in any branch, ties broken by the
// winning branch's score, then by id.
//
// Measured against the real corpus, this preserves each branch's champion
// where RRF k=60 diluted it: RRF rewards consensus, and these branches win
// by conviction on disjoint queries -- one model resolves a query the other
// two miss entirely.
func fuseMinRank(branches []rankedBranch) []int64 {
	type entry struct {
		id    int64
		rank  int
		score float64
	}
	best := make(map[int64]entry)
	for _, b := range branches {
		for pos, id := range b.ids {
			rank := pos + 1
			e, seen := best[id]
			if !seen || rank < e.rank || (rank == e.rank && b.score[id] > e.score) {
				best[id] = entry{id: id, rank: rank, score: b.score[id]}
			}
		}
	}
	out := make([]entry, 0, len(best))
	for _, e := range best {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].rank != out[j].rank {
			return out[i].rank < out[j].rank
		}
		if out[i].score != out[j].score {
			return out[i].score > out[j].score
		}
		return out[i].id < out[j].id
	})
	ids := make([]int64, len(out))
	for i, e := range out {
		ids[i] = e.id
	}
	return ids
}

// semanticCandidates ranks one model's stored vectors against the query
// vector, dropping anything under the floor. Brute force over all vectors:
// still microseconds at tens of thousands of observations.
func (s *Store) semanticCandidates(queryVector []float32, model string, opts SearchOptions, floor float64) (rankedBranch, error) {
	stored, err := s.LoadEmbeddings(model, opts)
	if err != nil {
		return rankedBranch{}, err
	}

	type scored struct {
		id  int64
		sim float64
	}
	hits := make([]scored, 0, len(stored))
	for _, e := range stored {
		if sim := cosine(queryVector, e.Vector); sim >= floor {
			hits = append(hits, scored{e.ID, sim})
		}
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].sim != hits[j].sim {
			return hits[i].sim > hits[j].sim
		}
		return hits[i].id < hits[j].id
	})

	b := rankedBranch{
		ids:   make([]int64, len(hits)),
		score: make(map[int64]float64, len(hits)),
	}
	for i, h := range hits {
		b.ids[i] = h.id
		b.score[h.id] = h.sim
	}
	return b, nil
}

// semanticBranches embeds the query with every model the embedder declares
// and returns one ranked branch per model. The second value is false when
// the semantic branch is unavailable (no ENGRAM_EMBEDDER, or the command
// failed). It never returns an error: a search must not break because a
// model did not start.
func (s *Store) semanticBranches(query string, opts SearchOptions) ([]rankedBranch, bool) {
	cmd := os.Getenv("ENGRAM_EMBEDDER")
	if cmd == "" {
		return nil, false // branch off by configuration, not a failure: stay quiet
	}

	// From here on the user asked for semantic search, so a failure is worth
	// reporting: it degrades results silently otherwise.
	e, err := s.sharedEmbedder(cmd)
	if err != nil {
		log.Printf("[engram] semantic search off: %v", err)
		return nil, false
	}

	ctx, cancel := context.WithTimeout(context.Background(), embedTimeout)
	defer cancel()

	res, err := e.Embed(ctx, []embed.Request{{ID: "q", Kind: embed.KindQuery, Text: query}})
	if err != nil {
		log.Printf("[engram] semantic search off: embedding the query failed: %v", err)
		return nil, false
	}

	floor := similarityFloor()
	var branches []rankedBranch
	for _, r := range res {
		if r.Err != nil {
			log.Printf("[engram] semantic branch %q skipped: %v", r.Model, r.Err)
			continue
		}
		if len(r.Vector) == 0 {
			log.Printf("[engram] semantic branch %q skipped: empty vector", r.Model)
			continue
		}
		b, err := s.semanticCandidates(r.Vector, r.Model, opts, floor)
		if err != nil {
			log.Printf("[engram] semantic branch %q skipped: %v", r.Model, err)
			continue
		}
		if len(b.ids) == 0 {
			log.Printf("[engram] semantic branch %q: no candidates above the %.2f floor "+
				"(are the vectors backfilled for this model?)", r.Model, floor)
			continue
		}
		branches = append(branches, b)
	}
	return branches, len(branches) > 0
}

// sharedEmbedder keeps one embedder per Store so the models load once per
// session, not once per search. It dies with the process.
func (s *Store) sharedEmbedder(cmd string) (embed.Embedder, error) {
	s.embMu.Lock()
	defer s.embMu.Unlock()
	if s.emb != nil {
		return s.emb, nil
	}
	e, err := embed.NewEmbedder(cmd)
	if err != nil {
		return nil, err
	}
	s.emb = e
	return e, nil
}

// observationsByIDs fetches full observations for ids the semantic branch
// contributed and the keyword branch had not already fetched.
func (s *Store) observationsByIDs(ids []int64) ([]Observation, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	placeholders := ""
	args := make([]any, 0, len(ids))
	for i, id := range ids {
		if i > 0 {
			placeholders += ","
		}
		placeholders += "?"
		args = append(args, id)
	}
	rows, err := s.queryItHook(s.db,
		`SELECT `+observationSelectColumns+`
		   FROM observations
		  WHERE id IN (`+placeholders+`) AND deleted_at IS NULL`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Observation
	for rows.Next() {
		var o Observation
		if err := rows.Scan(&o.ID, &o.SyncID, &o.SessionID, &o.Type, &o.Title, &o.Content,
			&o.ToolName, &o.Project, &o.Scope, &o.TopicKey, &o.RevisionCount, &o.DuplicateCount,
			&o.LastSeenAt, &o.ReviewAfter, &o.Pinned, &o.CreatedAt, &o.UpdatedAt, &o.DeletedAt); err != nil {
			return nil, err
		}
		out = append(out, o)
	}
	return out, rows.Err()
}
