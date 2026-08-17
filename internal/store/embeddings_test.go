package store

import (
	"math"
	"testing"
)

// obsForEmbedding stores one observation and returns its id. Observations
// have a foreign key to sessions, so the session is created first.
func obsForEmbedding(t *testing.T, s *Store, title, content string) int64 {
	t.Helper()
	if err := s.CreateSession("s1", "engram", "/tmp/engram"); err != nil {
		t.Fatalf("create session: %v", err)
	}
	id, err := s.AddObservation(AddObservationParams{
		SessionID: "s1",
		Type:      "manual",
		Title:     title,
		Content:   content,
		Project:   "engram",
		Scope:     "project",
	})
	if err != nil {
		t.Fatalf("AddObservation: %v", err)
	}
	return id
}

func TestVectorRoundTrip(t *testing.T) {
	orig := []float32{0.5, -0.25, 0, 1}
	got, err := unpackVector(packVector(orig))
	if err != nil {
		t.Fatalf("unpack: %v", err)
	}
	if len(got) != len(orig) {
		t.Fatalf("length %d, expected %d", len(got), len(orig))
	}
	for i := range orig {
		if math.Abs(float64(got[i]-orig[i])) > 1e-9 {
			t.Errorf("pos %d: %v, expected %v", i, got[i], orig[i])
		}
	}
}

func TestUnpackRejectsInvalidLength(t *testing.T) {
	if _, err := unpackVector([]byte{1, 2, 3}); err == nil {
		t.Fatal("expected an error for a blob that is not a multiple of 4")
	}
}

func TestSetEmbeddingUpsertsPerModel(t *testing.T) {
	s := newTestStore(t)
	id := obsForEmbedding(t, s, "a note", "some text")

	if err := s.SetEmbedding(id, "model-a", []float32{1, 0}); err != nil {
		t.Fatalf("SetEmbedding a: %v", err)
	}
	if err := s.SetEmbedding(id, "model-b", []float32{0, 1}); err != nil {
		t.Fatalf("SetEmbedding b: %v", err)
	}
	// Overwriting the same (observation, model) must not add a row.
	if err := s.SetEmbedding(id, "model-a", []float32{0.6, 0.8}); err != nil {
		t.Fatalf("SetEmbedding a again: %v", err)
	}

	a, err := s.LoadEmbeddings("model-a", SearchOptions{})
	if err != nil {
		t.Fatalf("LoadEmbeddings: %v", err)
	}
	if len(a) != 1 {
		t.Fatalf("model-a rows = %d, expected 1", len(a))
	}
	if a[0].ID != id || math.Abs(float64(a[0].Vector[0]-0.6)) > 1e-6 {
		t.Errorf("model-a = %+v, expected the updated vector for obs %d", a[0], id)
	}

	b, err := s.LoadEmbeddings("model-b", SearchOptions{})
	if err != nil {
		t.Fatalf("LoadEmbeddings b: %v", err)
	}
	if len(b) != 1 || b[0].Vector[1] != 1 {
		t.Fatalf("model-b rows = %+v, expected the {0,1} vector", b)
	}
}

func TestSetEmbeddingRejectsEmptyInput(t *testing.T) {
	s := newTestStore(t)
	id := obsForEmbedding(t, s, "a note", "some text")

	if err := s.SetEmbedding(id, "model-a", nil); err == nil {
		t.Error("expected an error for an empty vector")
	}
	if err := s.SetEmbedding(id, "", []float32{1, 0}); err == nil {
		t.Error("expected an error for an empty model name")
	}
}

func TestPendingEmbeddingsIsPerModel(t *testing.T) {
	s := newTestStore(t)
	id := obsForEmbedding(t, s, "a note", "some text")
	if err := s.SetEmbedding(id, "model-a", []float32{1, 0}); err != nil {
		t.Fatalf("SetEmbedding: %v", err)
	}

	pendA, err := s.PendingEmbeddings("model-a", 10)
	if err != nil {
		t.Fatalf("PendingEmbeddings a: %v", err)
	}
	for _, p := range pendA {
		if p.ID == id {
			t.Errorf("obs %d has a model-a vector but is still pending", id)
		}
	}

	pendB, err := s.PendingEmbeddings("model-b", 10)
	if err != nil {
		t.Fatalf("PendingEmbeddings b: %v", err)
	}
	found := false
	for _, p := range pendB {
		if p.ID == id {
			found = true
			if p.Title != "a note" || p.Content != "some text" {
				t.Errorf("pending row = %+v, expected the title and content back", p)
			}
		}
	}
	if !found {
		t.Errorf("obs %d has no model-b vector and is not pending for it", id)
	}
}

// A deleted observation must never be handed out for embedding, and its
// stored vectors must drop out of the search universe.
func TestDeletedObservationsAreExcluded(t *testing.T) {
	s := newTestStore(t)
	id := obsForEmbedding(t, s, "a note", "some text")
	if err := s.SetEmbedding(id, "model-a", []float32{1, 0}); err != nil {
		t.Fatalf("SetEmbedding: %v", err)
	}
	if _, err := s.execHook(s.db,
		`UPDATE observations SET deleted_at = datetime('now') WHERE id = ?`, id); err != nil {
		t.Fatalf("soft delete: %v", err)
	}

	loaded, err := s.LoadEmbeddings("model-a", SearchOptions{})
	if err != nil {
		t.Fatalf("LoadEmbeddings: %v", err)
	}
	for _, e := range loaded {
		if e.ID == id {
			t.Error("a deleted observation's vector is still in the search universe")
		}
	}

	pending, err := s.PendingEmbeddings("model-b", 10)
	if err != nil {
		t.Fatalf("PendingEmbeddings: %v", err)
	}
	for _, p := range pending {
		if p.ID == id {
			t.Error("a deleted observation is still queued for embedding")
		}
	}
}

// LoadEmbeddings must apply the same filters as the keyword branch, or the
// two branches would rank over different universes.
func TestLoadEmbeddingsAppliesSearchFilters(t *testing.T) {
	s := newTestStore(t)
	id := obsForEmbedding(t, s, "a note", "some text")
	if err := s.SetEmbedding(id, "model-a", []float32{1, 0}); err != nil {
		t.Fatalf("SetEmbedding: %v", err)
	}

	if got, err := s.LoadEmbeddings("model-a", SearchOptions{Project: "engram"}); err != nil {
		t.Fatalf("LoadEmbeddings: %v", err)
	} else if len(got) != 1 {
		t.Errorf("matching project returned %d rows, expected 1", len(got))
	}

	if got, err := s.LoadEmbeddings("model-a", SearchOptions{Project: "other-project"}); err != nil {
		t.Fatalf("LoadEmbeddings: %v", err)
	} else if len(got) != 0 {
		t.Errorf("non-matching project returned %d rows, expected 0", len(got))
	}

	if got, err := s.LoadEmbeddings("model-a", SearchOptions{Type: "bugfix"}); err != nil {
		t.Fatalf("LoadEmbeddings: %v", err)
	} else if len(got) != 0 {
		t.Errorf("non-matching type returned %d rows, expected 0", len(got))
	}
}

// Search normalizes the project filter before matching, so a caller that
// passes "Engram" must reach the same rows as one that passes "engram".
func TestLoadEmbeddingsNormalizesProject(t *testing.T) {
	s := newTestStore(t)
	id := obsForEmbedding(t, s, "a note", "some text")
	if err := s.SetEmbedding(id, "model-a", []float32{1, 0}); err != nil {
		t.Fatalf("SetEmbedding: %v", err)
	}

	got, err := s.LoadEmbeddings("model-a", SearchOptions{Project: "Engram"})
	if err != nil {
		t.Fatalf("LoadEmbeddings: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("project \"Engram\" returned %d rows, expected 1 (same as \"engram\")", len(got))
	}
}
