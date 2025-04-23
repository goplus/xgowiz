package ollama

import (
	"fmt"
	"strings"
	"time"

	"github.com/goplus/xgowiz/llm"
	api "github.com/ollama/ollama/api"
)

// OllamaMessage adapts Ollama's message format to our Message interface
type OllamaMessage struct {
	Message    api.Message
	ToolCallID string // Store tool call ID separately since Ollama API doesn't have this field
}

func (m *OllamaMessage) Role() string {
	return m.Message.Role
}

func (m *OllamaMessage) Content() string {
	// For tool responses and regular messages, just return the content string
	return strings.TrimSpace(m.Message.Content)
}

func (m *OllamaMessage) ToolCalls() []llm.ToolCall {
	var calls []llm.ToolCall
	for _, call := range m.Message.ToolCalls {
		calls = append(calls, NewOllamaToolCall(call))
	}
	return calls
}

func (m *OllamaMessage) StatUsage() (int, int) {
	return 0, 0 // Ollama doesn't provide token usage info
}

func (m *OllamaMessage) ToolResponse() (string, bool) {
	return m.ToolCallID, m.Message.Role == "tool"
}

// OllamaToolCall adapts Ollama's tool call format
type OllamaToolCall struct {
	call api.ToolCall
	id   string // Store a unique ID for the tool call
}

func NewOllamaToolCall(call api.ToolCall) *OllamaToolCall {
	return &OllamaToolCall{
		call: call,
		id: fmt.Sprintf(
			"tc_%s_%d",
			call.Function.Name,
			time.Now().UnixNano(),
		),
	}
}

func (t *OllamaToolCall) Name() string {
	return t.call.Function.Name
}

func (t *OllamaToolCall) Arguments() map[string]any {
	return t.call.Function.Arguments
}

func (t *OllamaToolCall) ID() string {
	return t.id
}
