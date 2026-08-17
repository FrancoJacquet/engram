// Package embed generates text vectors by shelling out to an external
// command, following the same pattern internal/llm uses for ENGRAM_AGENT_CLI.
// Engram does not bundle models, does not call APIs and does not manage keys.
// The command owns the models: it declares them in its handshake and applies
// each model's prompt template, so switching models never touches Go code.
package embed

import (
	"context"
	"errors"
)

// ErrNoEmbedder means ENGRAM_EMBEDDER is not configured. It is not a
// failure: it is the signal that the semantic branch is switched off.
var ErrNoEmbedder = errors.New("no embedder configured")

// Kind separates documents from queries. Embedding models use different
// trained prompts for each, and mixing them degrades quality without
// failing. Engram never sees the prompts: the command applies them.
const (
	KindPassage = "passage"
	KindQuery   = "query"
)

// ModelInfo describes one model the embedder command declared in its
// handshake.
type ModelInfo struct {
	Name string
	Dim  int
}

// Request is one text to vectorize. Title travels separately because some
// models have a dedicated title slot in their document template.
type Request struct {
	ID    string
	Kind  string // KindPassage | KindQuery
	Title string
	Text  string
}

// Result is one model's vector for one Request. Err set means that model
// failed for that text; other results in the batch remain valid.
type Result struct {
	ID     string
	Vector []float32
	Model  string
	Err    error
}

// Embedder vectorizes batches of text with every model the command runs.
// Embed returns len(reqs) x len(models) results, grouped by request in
// input order. Implementations may keep a child process alive between
// calls so models load once per session.
type Embedder interface {
	Models(ctx context.Context) ([]ModelInfo, error)
	Embed(ctx context.Context, reqs []Request) ([]Result, error)
	Close() error
}
