package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/goplus/xgowiz/llm"
	"github.com/qiniu/x/log"
)

type CreateRequest struct {
	Model     string         `json:"model"`
	Messages  []MessageParam `json:"messages"`
	MaxTokens int            `json:"max_tokens"`
	Tools     []Tool         `json:"tools,omitempty"`
}

type MessageParam struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

type ContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Content   any             `json:"content,omitempty"`
}

type Tool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema InputSchema `json:"input_schema"`
}

type InputSchema struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Required   []string       `json:"required,omitempty"`
}

type APIMessage struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []ContentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   *string        `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence"`
	Usage        Usage          `json:"usage"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Message implements the llm.Message interface
type Message struct {
	Msg APIMessage
}

func (m *Message) Role() string {
	return m.Msg.Role
}

func (m *Message) Content() string {
	log.Debug("getting content from message", "message", m)

	var content []string
	for i, block := range m.Msg.Content {
		log.Debug("processing content block", "index", i, "block", block)

		if block.Type == "text" {
			log.Debug("adding text block", "text", block.Text)
			content = append(content, block.Text)
		} else if block.Type == "tool_result" {
			log.Debug("processing tool result block", "block", block)

			// Handle the content directly if it's a string
			if contentStr, ok := block.Content.(string); ok {
				content = append(content, contentStr)
				continue
			}

			// Handle array of maps structure
			if contentArray, ok := block.Content.([]any); ok {
				for _, item := range contentArray {
					if contentMap, ok := item.(map[string]any); ok {
						if text, ok := contentMap["text"]; ok {
							textStr := fmt.Sprintf("%v", text)
							log.Debug("extracted text from content map", "text", textStr)
							content = append(content, textStr)
						}
					} else {
						// If it's not a map, try to convert it directly to string
						textStr := fmt.Sprintf("%v", item)
						log.Debug("extracted direct content", "text", textStr)
						content = append(content, textStr)
					}
				}
			}

			// If we still haven't found content and have Text field, use it
			if len(content) == 0 && block.Text != "" {
				log.Debug("falling back to direct text", "text", block.Text)
				content = append(content, block.Text)
			}
		}
	}

	result := strings.TrimSpace(strings.Join(content, " "))
	log.Debug("final content result", "content", result)
	return result
}

func (m *Message) ToolCalls() []llm.ToolCall {
	var calls []llm.ToolCall
	for _, block := range m.Msg.Content {
		if block.Type == "tool_use" {
			calls = append(calls, &ToolCall{
				id:   block.ID,
				name: block.Name,
				args: block.Input,
			})
		}
	}
	return calls
}

func (m *Message) ToolResponse() (toolCallID string, ok bool) {
	for _, block := range m.Msg.Content {
		if block.Type == "tool_result" {
			return block.ToolUseID, true
		}
	}
	return
}

func (m *Message) StatUsage() (input int, output int) {
	return m.Msg.Usage.InputTokens, m.Msg.Usage.OutputTokens
}

// ToolCall implements the llm.ToolCall interface
type ToolCall struct {
	id   string
	name string
	args json.RawMessage
}

func (t *ToolCall) Name() string {
	return t.name
}

func (t *ToolCall) Arguments() map[string]any {
	var args map[string]any
	if err := json.Unmarshal(t.args, &args); err != nil {
		return make(map[string]any)
	}
	return args
}

func (t *ToolCall) ID() string {
	return t.id
}
