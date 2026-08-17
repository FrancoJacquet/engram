package embed

import (
	"context"
	"testing"
)

// fakeCmd speaks the rev 2 NDJSON protocol without needing Python: a
// two-model handshake, then two response lines per input line.
const fakeCmd = `sh -c 'echo "{\"ready\":true,\"models\":[{\"name\":\"fakeA\",\"dim\":2},{\"name\":\"fakeB\",\"dim\":2}]}"; while read -r line; do id=$(printf %s "$line" | sed -n "s/.*\"id\":\"\\([^\"]*\\)\".*/\\1/p"); echo "{\"id\":\"$id\",\"embedding\":[1.0,0.0],\"model\":\"fakeA\",\"dim\":2}"; echo "{\"id\":\"$id\",\"embedding\":[0.0,1.0],\"model\":\"fakeB\",\"dim\":2}"; done'`

func TestModelsComeFromHandshake(t *testing.T) {
	e, err := NewEmbedder(fakeCmd)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()

	models, err := e.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 2 || models[0].Name != "fakeA" || models[1].Name != "fakeB" {
		t.Fatalf("models = %+v, expected fakeA and fakeB", models)
	}
	if models[0].Dim != 2 {
		t.Errorf("fakeA dim = %d, expected 2", models[0].Dim)
	}
}

func TestEmbedReturnsOneResultPerModelPerRequest(t *testing.T) {
	e, err := NewEmbedder(fakeCmd)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()

	got, err := e.Embed(context.Background(), []Request{
		{ID: "1", Kind: KindQuery, Text: "hello"},
		{ID: "2", Kind: KindPassage, Title: "a title", Text: "bye"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("expected 4 results (2 reqs x 2 models), got %d", len(got))
	}
	wantIDs := []string{"1", "1", "2", "2"}
	wantModels := []string{"fakeA", "fakeB", "fakeA", "fakeB"}
	for i := range got {
		if got[i].ID != wantIDs[i] || got[i].Model != wantModels[i] {
			t.Errorf("result %d = (%q, %q), expected (%q, %q)",
				i, got[i].ID, got[i].Model, wantIDs[i], wantModels[i])
		}
	}
}

func TestProcessIsReusedAcrossCalls(t *testing.T) {
	e, err := NewEmbedder(fakeCmd)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()

	for i := 0; i < 3; i++ {
		if _, err := e.Embed(context.Background(), []Request{{ID: "x", Kind: KindQuery, Text: "t"}}); err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}
}

func TestMissingCommandFails(t *testing.T) {
	e, err := NewEmbedder("/does/not/exist")
	if err == nil {
		_, err = e.Embed(context.Background(), []Request{{ID: "1", Kind: KindQuery, Text: "t"}})
		e.Close()
	}
	if err == nil {
		t.Fatal("expected an error, got none")
	}
}

// A per-text failure must not poison the rest of the batch: the command
// reports it inline and keeps going, so Embed returns it as Result.Err.
func TestPerTextErrorIsReportedNotFatal(t *testing.T) {
	const cmd = `sh -c 'echo "{\"ready\":true,\"models\":[{\"name\":\"fakeA\",\"dim\":2}]}"; while read -r line; do id=$(printf %s "$line" | sed -n "s/.*\"id\":\"\\([^\"]*\\)\".*/\\1/p"); echo "{\"id\":\"$id\",\"model\":\"fakeA\",\"error\":\"empty text\"}"; done'`

	e, err := NewEmbedder(cmd)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()

	got, err := e.Embed(context.Background(), []Request{{ID: "1", Kind: KindPassage, Text: ""}})
	if err != nil {
		t.Fatalf("Embed must not fail because one text did: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 result, got %d", len(got))
	}
	if got[0].Err == nil {
		t.Error("expected Result.Err to carry the per-text failure")
	}
	if len(got[0].Vector) != 0 {
		t.Errorf("expected no vector alongside an error, got %d floats", len(got[0].Vector))
	}
}
