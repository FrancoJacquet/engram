package embed

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
)

// processEmbedder keeps a child process alive with the models loaded.
// It starts on first use and dies on Close (or with the parent process).
type processEmbedder struct {
	command string

	mu     sync.Mutex
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	models []ModelInfo
}

type wireModel struct {
	Name string `json:"name"`
	Dim  int    `json:"dim"`
}

type wireResponse struct {
	Ready     bool        `json:"ready"`
	Models    []wireModel `json:"models"`
	ID        string      `json:"id"`
	Embedding []float32   `json:"embedding"`
	Model     string      `json:"model"`
	Error     string      `json:"error"`
}

func newProcessEmbedder(command string) (Embedder, error) {
	return &processEmbedder{command: command}, nil
}

// start launches the process and consumes the handshake line, which
// declares the models. Caller must hold the lock.
func (p *processEmbedder) start() error {
	if p.cmd != nil {
		return nil
	}
	cmd := exec.Command("sh", "-c", p.command)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("embedder stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("embedder stdout: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting embedder: %w", err)
	}
	reader := bufio.NewReaderSize(stdout, 1<<20) // 1 MB: 1024 floats per line

	line, err := reader.ReadBytes('\n')
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return fmt.Errorf("embedder did not greet: %w", err)
	}
	var ready wireResponse
	if err := json.Unmarshal(line, &ready); err != nil || !ready.Ready || len(ready.Models) == 0 {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return fmt.Errorf("invalid embedder handshake: %s", line)
	}

	models := make([]ModelInfo, len(ready.Models))
	for i, m := range ready.Models {
		models[i] = ModelInfo{Name: m.Name, Dim: m.Dim}
	}
	p.cmd, p.stdin, p.stdout, p.models = cmd, stdin, reader, models
	return nil
}

func (p *processEmbedder) Models(ctx context.Context) ([]ModelInfo, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.start(); err != nil {
		return nil, err
	}
	out := make([]ModelInfo, len(p.models))
	copy(out, p.models)
	return out, nil
}

func (p *processEmbedder) Embed(ctx context.Context, reqs []Request) ([]Result, error) {
	if len(reqs) == 0 {
		return nil, nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	if err := p.start(); err != nil {
		return nil, err
	}

	enc := json.NewEncoder(p.stdin)
	for _, r := range reqs {
		if r.Kind == "" {
			r.Kind = KindPassage
		}
		payload := map[string]string{"id": r.ID, "kind": r.Kind, "title": r.Title, "text": r.Text}
		if err := enc.Encode(payload); err != nil {
			p.killLocked()
			return nil, fmt.Errorf("writing to embedder: %w", err)
		}
	}

	// The protocol guarantees len(models) lines per request, errors included.
	want := len(reqs) * len(p.models)
	out := make([]Result, 0, want)
	for i := 0; i < want; i++ {
		if err := ctx.Err(); err != nil {
			p.killLocked()
			return nil, err
		}
		line, err := p.stdout.ReadBytes('\n')
		if err != nil {
			p.killLocked()
			return nil, fmt.Errorf("reading from embedder: %w", err)
		}
		var resp wireResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			p.killLocked()
			return nil, fmt.Errorf("unreadable embedder response: %w", err)
		}
		res := Result{ID: resp.ID, Vector: resp.Embedding, Model: resp.Model}
		if resp.Error != "" {
			res.Err = fmt.Errorf("embedder: %s", resp.Error)
		}
		out = append(out, res)
	}
	return out, nil
}

func (p *processEmbedder) killLocked() {
	if p.cmd != nil && p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
		_ = p.cmd.Wait()
	}
	p.cmd, p.stdin, p.stdout, p.models = nil, nil, nil, nil
}

func (p *processEmbedder) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cmd == nil {
		return nil
	}
	_ = p.stdin.Close()
	err := p.cmd.Wait()
	p.cmd, p.stdin, p.stdout, p.models = nil, nil, nil, nil
	return err
}
