package store

import "testing"

// Saving a memory must never fail or stall because of the models: the
// embedding is best-effort and the next backfill picks up what is missing.
func TestSaveSucceedsEvenIfEmbedderExplodes(t *testing.T) {
	t.Setenv("ENGRAM_EMBEDDER", "sh -c 'exit 1'")

	s := newTestStore(t)
	if err := s.CreateSession("s1", "engram", "/tmp/engram"); err != nil {
		t.Fatalf("create session: %v", err)
	}
	id, err := s.AddObservation(AddObservationParams{
		SessionID: "s1", Type: "manual",
		Title: "a title", Content: "some content",
		Project: "engram", Scope: "project",
	})
	if err != nil {
		t.Fatalf("AddObservation failed with a broken embedder: %v", err)
	}
	if id == 0 {
		t.Fatal("the observation was not stored")
	}
}

func TestSaveSucceedsWithNoEmbedderConfigured(t *testing.T) {
	t.Setenv("ENGRAM_EMBEDDER", "")

	s := newTestStore(t)
	if err := s.CreateSession("s1", "engram", "/tmp/engram"); err != nil {
		t.Fatalf("create session: %v", err)
	}
	if _, err := s.AddObservation(AddObservationParams{
		SessionID: "s1", Type: "manual",
		Title: "a title", Content: "some content",
		Project: "engram", Scope: "project",
	}); err != nil {
		t.Fatalf("AddObservation: %v", err)
	}
}
