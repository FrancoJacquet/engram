package embed

import "fmt"

// NewEmbedder returns an Embedder that shells out to the given command.
// The command comes from ENGRAM_EMBEDDER; empty returns ErrNoEmbedder,
// which callers must treat as "semantic branch off", not as a failure.
//
//	e, err := embed.NewEmbedder(os.Getenv("ENGRAM_EMBEDDER"))
//	if errors.Is(err, embed.ErrNoEmbedder) { /* keyword-only search */ }
func NewEmbedder(command string) (Embedder, error) {
	if command == "" {
		return nil, fmt.Errorf("%w: ENGRAM_EMBEDDER is not set", ErrNoEmbedder)
	}
	return newProcessEmbedder(command)
}
