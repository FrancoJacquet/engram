package store

import (
	"math"
	"testing"
)

func TestCosineOfNormalizedVectors(t *testing.T) {
	a := []float32{1, 0}
	if got := cosine(a, []float32{1, 0}); math.Abs(got-1) > 1e-6 {
		t.Errorf("identical -> %v, expected 1", got)
	}
	if got := cosine(a, []float32{0, 1}); math.Abs(got) > 1e-6 {
		t.Errorf("orthogonal -> %v, expected 0", got)
	}
	if got := cosine(a, []float32{-1, 0}); math.Abs(got+1) > 1e-6 {
		t.Errorf("opposite -> %v, expected -1", got)
	}
}

// Vectors from different models have different lengths and are not
// comparable; returning 0 keeps one stale row from breaking a search.
func TestCosineWithMismatchedLengthsIsZero(t *testing.T) {
	if got := cosine([]float32{1, 0}, []float32{1, 0, 0}); got != 0 {
		t.Errorf("mismatched lengths -> %v, expected 0", got)
	}
	if got := cosine(nil, nil); got != 0 {
		t.Errorf("empty vectors -> %v, expected 0", got)
	}
}

func TestMinRankKeepsEachBranchChampion(t *testing.T) {
	keyword := rankedBranch{
		ids:   []int64{10, 20},
		score: map[int64]float64{10: 2.0, 20: 1.5},
	}
	dense := rankedBranch{
		ids:   []int64{30, 10},
		score: map[int64]float64{30: 0.9, 10: 0.5},
	}
	got := fuseMinRank([]rankedBranch{keyword, dense})

	// 10 and 30 are both rank 1 in some branch; the tie breaks by the
	// winning branch's score (2.0 > 0.9). 20 is rank 2. No duplicates.
	want := []int64{10, 30, 20}
	if len(got) != len(want) {
		t.Fatalf("got %v, expected %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, expected %v", got, want)
		}
	}
}

func TestMinRankWithOneEmptyBranchKeepsTheOther(t *testing.T) {
	got := fuseMinRank([]rankedBranch{
		{ids: []int64{7, 8}, score: map[int64]float64{7: 1, 8: 0.5}},
		{},
	})
	if len(got) != 2 || got[0] != 7 || got[1] != 8 {
		t.Errorf("got %v, expected [7 8]", got)
	}
}

// The whole point of min-rank over RRF: a document that wins in one branch
// alone must keep that position instead of being diluted by the branches
// that did not find it.
func TestMinRankPrefersTheBestRankAcrossBranches(t *testing.T) {
	a := rankedBranch{ids: []int64{1, 2, 40}, score: map[int64]float64{1: 9, 2: 8, 40: 7}}
	b := rankedBranch{ids: []int64{40}, score: map[int64]float64{40: 0.4}}
	got := fuseMinRank([]rankedBranch{a, b})

	// 40 is rank 3 in branch A but rank 1 in branch B: min-rank uses 1, so
	// it ties with 1 and loses the tie only on score (9 > 0.4). 2 stays last.
	want := []int64{1, 40, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, expected %v", got, want)
		}
	}
}

func TestMinRankWithNoBranchesIsEmpty(t *testing.T) {
	if got := fuseMinRank(nil); len(got) != 0 {
		t.Errorf("got %v, expected no ids", got)
	}
}

// semanticCandidates must drop anything under the floor and rank the rest
// by similarity.
func TestSemanticCandidatesRanksAndAppliesFloor(t *testing.T) {
	s := newTestStore(t)
	near := obsForEmbedding(t, s, "near", "near")
	far := obsForEmbedding(t, s, "far", "far")

	// Unit vectors: near is identical to the query, far is orthogonal.
	if err := s.SetEmbedding(near, "m", []float32{1, 0}); err != nil {
		t.Fatalf("SetEmbedding near: %v", err)
	}
	if err := s.SetEmbedding(far, "m", []float32{0, 1}); err != nil {
		t.Fatalf("SetEmbedding far: %v", err)
	}

	b, err := s.semanticCandidates([]float32{1, 0}, "m", SearchOptions{}, 0.5)
	if err != nil {
		t.Fatalf("semanticCandidates: %v", err)
	}
	if len(b.ids) != 1 || b.ids[0] != near {
		t.Fatalf("got %v, expected only the near observation (%d)", b.ids, near)
	}
	if math.Abs(b.score[near]-1) > 1e-6 {
		t.Errorf("score = %v, expected 1 for an identical vector", b.score[near])
	}
}

func TestSearchUnchangedWithoutEmbedder(t *testing.T) {
	t.Setenv("ENGRAM_EMBEDDER", "")

	s := newTestStore(t)
	if err := s.CreateSession("s1", "engram", "/tmp/engram"); err != nil {
		t.Fatalf("create session: %v", err)
	}
	if _, err := s.AddObservation(AddObservationParams{
		SessionID: "s1", Type: "manual",
		Title:   "the database tunnel went down",
		Content: "ERROR 2013 Lost connection to MySQL server",
		Project: "engram", Scope: "project",
	}); err != nil {
		t.Fatalf("AddObservation: %v", err)
	}

	byKeyword, err := s.Search("tunnel", SearchOptions{Limit: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(byKeyword) != 1 {
		t.Fatalf("expected 1 keyword hit, got %d", len(byKeyword))
	}

	// With no embedder there is no semantic branch: a paraphrase finds nothing.
	byParaphrase, err := s.Search("cannot reach the customer database", SearchOptions{Limit: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(byParaphrase) != 0 {
		t.Errorf("without an embedder expected 0 results, got %d", len(byParaphrase))
	}
}

// A broken embedder must degrade to keyword-only results, never fail the
// search: ENGRAM_EMBEDDER pointing at a command that dies is the realistic
// failure mode (bad path, missing venv, model that will not load).
func TestSearchFallsBackWhenEmbedderIsBroken(t *testing.T) {
	t.Setenv("ENGRAM_EMBEDDER", "sh -c 'exit 1'")

	s := newTestStore(t)
	if err := s.CreateSession("s1", "engram", "/tmp/engram"); err != nil {
		t.Fatalf("create session: %v", err)
	}
	if _, err := s.AddObservation(AddObservationParams{
		SessionID: "s1", Type: "manual",
		Title:   "the database tunnel went down",
		Content: "ERROR 2013 Lost connection to MySQL server",
		Project: "engram", Scope: "project",
	}); err != nil {
		t.Fatalf("AddObservation: %v", err)
	}

	got, err := s.Search("tunnel", SearchOptions{Limit: 10})
	if err != nil {
		t.Fatalf("a broken embedder must not fail the search: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("expected the keyword hit to survive, got %d results", len(got))
	}
}

// The semantic branch must contribute observations FTS5 never matched --
// that is the entire point -- and they must come back fully populated, not
// as bare ids.
func TestSemanticBranchAddsResultsKeywordSearchMissed(t *testing.T) {
	s := newTestStore(t)
	if err := s.CreateSession("s1", "engram", "/tmp/engram"); err != nil {
		t.Fatalf("create session: %v", err)
	}
	id, err := s.AddObservation(AddObservationParams{
		SessionID: "s1", Type: "manual",
		Title:   "the database tunnel went down",
		Content: "ERROR 2013 Lost connection to MySQL server",
		Project: "engram", Scope: "project",
	})
	if err != nil {
		t.Fatalf("AddObservation: %v", err)
	}

	// A fake embedder that always returns the same unit vector, so the
	// stored vector below is a perfect match for any query.
	t.Setenv("ENGRAM_EMBEDDER", `sh -c 'echo "{\"ready\":true,\"models\":[{\"name\":\"m\",\"dim\":2}]}"; while read -r line; do id=$(printf %s "$line" | sed -n "s/.*\"id\":\"\\([^\"]*\\)\".*/\\1/p"); echo "{\"id\":\"$id\",\"embedding\":[1.0,0.0],\"model\":\"m\",\"dim\":2}"; done'`)
	if err := s.SetEmbedding(id, "m", []float32{1, 0}); err != nil {
		t.Fatalf("SetEmbedding: %v", err)
	}

	// No shared tokens with the observation: FTS5 alone returns nothing.
	got, err := s.Search("cannot reach the customer database", SearchOptions{Limit: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected the semantic branch to contribute 1 result, got %d", len(got))
	}
	if got[0].ID != id {
		t.Errorf("got observation %d, expected %d", got[0].ID, id)
	}
	if got[0].Title == "" || got[0].Content == "" {
		t.Errorf("semantic result came back unpopulated: %+v", got[0])
	}
}
