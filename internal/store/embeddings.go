package store

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// PendingEmbedding is an observation that needs (re)embedding for a model.
type PendingEmbedding struct {
	ID      int64
	Title   string
	Content string
}

// StoredEmbedding is a stored vector, ready to compare.
type StoredEmbedding struct {
	ID     int64
	Vector []float32
}

// packVector serializes float32 little-endian (768 dims = 3072 bytes).
func packVector(v []float32) []byte {
	out := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(f))
	}
	return out
}

func unpackVector(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("blob of %d bytes: not a multiple of 4", len(b))
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}

// SetEmbedding stores the vector one model produced for one observation.
// One row per (observation, model): vectors from different models are
// incomparable, and the model key is what makes re-embedding detectable.
func (s *Store) SetEmbedding(id int64, model string, vector []float32) error {
	if len(vector) == 0 {
		return fmt.Errorf("empty vector for observation %d", id)
	}
	if strings.TrimSpace(model) == "" {
		return fmt.Errorf("empty model for observation %d", id)
	}
	_, err := s.execHook(s.db,
		`INSERT INTO observation_embeddings (observation_id, model, embedding, created_at)
		 VALUES (?, ?, ?, datetime('now'))
		 ON CONFLICT(observation_id, model) DO UPDATE SET
		   embedding = excluded.embedding, created_at = excluded.created_at`,
		id, model, packVector(vector))
	return err
}

// PendingEmbeddings returns observations that need embedding under the
// given model: rows with no vector for it, and rows edited after they
// were embedded (embedding older than the observation's updated_at).
func (s *Store) PendingEmbeddings(model string, limit int) ([]PendingEmbedding, error) {
	if limit <= 0 {
		limit = 500
	}
	rows, err := s.queryItHook(s.db,
		`SELECT o.id, o.title, o.content
		   FROM observations o
		   LEFT JOIN observation_embeddings e
		     ON e.observation_id = o.id AND e.model = ?
		  WHERE o.deleted_at IS NULL
		    AND (e.observation_id IS NULL OR e.created_at < o.updated_at)
		  ORDER BY o.id
		  LIMIT ?`, model, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []PendingEmbedding
	for rows.Next() {
		var p PendingEmbedding
		if err := rows.Scan(&p.ID, &p.Title, &p.Content); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// LoadEmbeddings returns the vectors of one model, applying the same
// filters as the keyword branch so both branches rank over the exact same
// universe of observations. The project filter is normalized here as well
// as in Search, so callers that reach this directly behave identically.
func (s *Store) LoadEmbeddings(model string, opts SearchOptions) ([]StoredEmbedding, error) {
	project, _ := NormalizeProject(opts.Project)

	q := strings.Builder{}
	q.WriteString(`SELECT e.observation_id, e.embedding
	                 FROM observation_embeddings e
	                 JOIN observations o ON o.id = e.observation_id
	                WHERE o.deleted_at IS NULL AND e.model = ?`)
	args := []any{model}

	if opts.Type != "" {
		q.WriteString(" AND o.type = ?")
		args = append(args, opts.Type)
	}
	if project != "" {
		q.WriteString(" AND LOWER(o.project) = ?")
		args = append(args, project)
	}
	if opts.Scope != "" {
		q.WriteString(" AND o.scope = ?")
		args = append(args, normalizeScope(opts.Scope))
	}

	rows, err := s.queryItHook(s.db, q.String(), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []StoredEmbedding
	for rows.Next() {
		var id int64
		var blob []byte
		if err := rows.Scan(&id, &blob); err != nil {
			return nil, err
		}
		vec, err := unpackVector(blob)
		if err != nil {
			continue // one corrupt blob must not break the whole search
		}
		out = append(out, StoredEmbedding{ID: id, Vector: vec})
	}
	return out, rows.Err()
}
