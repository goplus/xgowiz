package history

import (
	"encoding/json"
	"strings"

	"github.com/goplus/xgowiz/llm"
)

// HistoryMessage implements the llm.Message interface for stored messages
type HistoryMessage struct {
	ARole    string         `json:"role"`
	AContent []ContentBlock `json:"content"`
}

func (m *HistoryMessage) Role() string {
	return m.ARole
}

func (m *HistoryMessage) Content() string {
	// Concatenate all text content blocks
	var content string
	for _, block := range m.AContent { // TODO(xsw)
		if block.Type == "text" {
			content += block.Text + " "
		}
	}
	return strings.TrimSpace(content)
}

func (m *HistoryMessage) ToolCalls() []llm.ToolCall {
	var calls []llm.ToolCall
	for _, block := range m.AContent {
		if block.Type == "tool_use" {
			calls = append(calls, &HistoryToolCall{
				id:   block.ID,
				name: block.Name,
				args: block.Input,
			})
		}
	}
	return calls
}

func (m *HistoryMessage) ToolResponse() (id string, ok bool) {
	for _, block := range m.AContent {
		if block.Type == "tool_result" {
			return block.ToolUseID, true
		}
	}
	return
}

func (m *HistoryMessage) StatUsage() (int, int) {
	return 0, 0 // History doesn't track usage
}

// HistoryToolCall implements llm.ToolCall for stored tool calls
type HistoryToolCall struct {
	id   string
	name string
	args json.RawMessage
}

func (t *HistoryToolCall) ID() string {
	return t.id
}

func (t *HistoryToolCall) Name() string {
	return t.name
}

func (t *HistoryToolCall) Arguments() map[string]any {
	var args map[string]any
	if err := json.Unmarshal(t.args, &args); err != nil {
		return make(map[string]any)
	}
	return args
}

// ContentBlock represents a block of content in a message
type ContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Content   interface{}     `json:"content,omitempty"`
}
