package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"

	"github.com/goplus/xgowiz/llm"
	"github.com/goplus/xgowiz/llm/history"
	"github.com/qiniu/x/log"
)

var (
	_ llm.Provider = (*Provider)(nil)
)

type Provider struct {
	client Client
	model  string
}

func convertSchema(schema llm.Schema) map[string]any {
	// Ensure required is a valid array, defaulting to empty if nil
	required := schema.Required
	if required == nil {
		required = []string{}
	}

	return map[string]any{
		"type":       schema.Type,
		"properties": schema.Properties,
		"required":   required,
	}
}

func NewProvider(apiKey string, baseURL string, client *http.Client, model string) *Provider {
	ret := &Provider{
		model: model,
	}
	ret.client.Init(apiKey, baseURL, client)
	return ret
}

func (p *Provider) SendMessage(ctx context.Context, prompt string, messages []llm.Message, tools []llm.Tool) (llm.Message, error) {
	log.Debug("creating message",
		"prompt", prompt,
		"num_messages", len(messages),
		"num_tools", len(tools))

	openaiMessages := make([]MessageParam, 0, len(messages))

	// Convert previous messages
	for _, msg := range messages {
		log.Debug("converting message",
			"role", msg.Role(),
			"content", msg.Content(),
			"is_tool_response", llm.IsToolResponse(msg))

		param := MessageParam{
			Role: msg.Role(),
		}

		if content := msg.Content(); content != "" {
			param.Content = &content
		}

		// Handle function/tool calls
		toolCalls := msg.ToolCalls()
		if len(toolCalls) > 0 {
			param.Content = nil // Must be null for function calls

			// Convert to OpenAI tool calls format
			param.ToolCalls = make([]ToolCall, len(toolCalls))
			for i, call := range toolCalls {
				args, err := json.Marshal(call.Arguments())
				if err != nil {
					return nil, fmt.Errorf(
						"error marshaling function arguments: %w",
						err,
					)
				}

				param.ToolCalls[i] = ToolCall{
					ID:   call.ID(),
					Type: "function",
					Function: FunctionCall{
						Name:      call.Name(),
						Arguments: string(args),
					},
				}
			}
		}

		// Handle function/tool responses
		if toolCallID, ok := msg.ToolResponse(); ok {
			log.Debug("processing tool response",
				"tool_call_id", toolCallID,
				"raw_message", msg)

			// Extract content from tool response
			var contentStr string
			if content := msg.Content(); content != "" {
				contentStr = content
			} else {
				// Try to extract text from history message content blocks
				if historyMsg, ok := msg.(*history.HistoryMessage); ok {
					var texts []string
					for _, block := range historyMsg.AContent {
						if block.Type == "tool_result" {
							if block.Text != "" {
								texts = append(texts, block.Text)
							} else if contentArray, ok := block.Content.([]any); ok {
								for _, item := range contentArray {
									if contentMap, ok := item.(map[string]any); ok {
										if text, ok := contentMap["text"]; ok {
											texts = append(texts, fmt.Sprint(text))
										}
									}
								}
							}
						}
					}
					contentStr = strings.Join(texts, "\n")
				}
			}

			if contentStr == "" {
				contentStr = "No content returned from function"
			}

			param.Content = &contentStr
			param.Role = "tool" // Use tool role instead of function
			param.ToolCallID = toolCallID
			// Don't set name field for tool responses
		}

		openaiMessages = append(openaiMessages, param)
	}

	// Log the final message array
	log.Debug("sending messages to OpenAI",
		"messages", openaiMessages,
		"num_tools", len(tools))

	// Add the new prompt if provided
	if prompt != "" {
		content := prompt
		openaiMessages = append(openaiMessages, MessageParam{
			Role:    "user",
			Content: &content,
		})
	}

	// Convert tools to OpenAI format
	openaiTools := make([]Tool, len(tools))
	for i, tool := range tools {
		openaiTools[i] = Tool{
			Type: "function",
			Function: FunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  convertSchema(tool.InputSchema),
			},
		}
	}

	// Make the API call
	resp, err := p.client.CreateChatCompletion(ctx, CreateRequest{
		Model:       p.model,
		Messages:    openaiMessages,
		Tools:       openaiTools,
		MaxTokens:   4096,
		Temperature: 0.7,
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	return &Message{Resp: resp, Choice: &resp.Choices[0]}, nil
}

func (p *Provider) SupportsTools() bool {
	return true
}

func (p *Provider) Name() string {
	return "openai"
}

func (p *Provider) CreateToolResponse(toolCallID string, content any) (llm.Message, error) {
	log.Debug("creating tool response",
		"tool_call_id", toolCallID,
		"content_type", reflect.TypeOf(content),
		"content", content)

	// Convert content to string representation
	var contentStr string
	switch v := content.(type) {
	case string:
		contentStr = v
	case []any:
		// Handle array of content blocks
		var texts []string
		for _, item := range v {
			if block, ok := item.(map[string]any); ok {
				// First try to get text directly
				if text, ok := block["text"].(string); ok {
					texts = append(texts, text)
					continue
				}

				// Then try array of text
				if textArray, ok := block["text"].([]any); ok {
					for _, t := range textArray {
						if str, ok := t.(string); ok {
							texts = append(texts, str)
						}
					}
					continue
				}

				// Finally try nested content array
				if contentArray, ok := block["content"].([]any); ok {
					for _, c := range contentArray {
						if cMap, ok := c.(map[string]any); ok {
							if text, ok := cMap["text"].(string); ok {
								texts = append(texts, text)
							}
						}
					}
				}
			}
		}
		contentStr = strings.Join(texts, "\n")
		if contentStr == "" {
			// Fallback to JSON if no text found
			jsonBytes, err := json.Marshal(v)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal array content: %w", err)
			}
			contentStr = string(jsonBytes)
		}
	default:
		// For other types, marshal to JSON
		jsonBytes, err := json.Marshal(content)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal tool response: %w", err)
		}
		contentStr = string(jsonBytes)
	}

	if contentStr == "" {
		contentStr = "No content returned from tool"
	}

	// Create a new message with the tool response
	msg := &Message{
		Choice: &Choice{
			Message: MessageParam{
				Role:       "tool",
				Content:    &contentStr,
				ToolCallID: toolCallID,
			},
		},
	}

	// Also set the response field
	msg.Resp = &APIResponse{
		Choices: []Choice{*msg.Choice},
	}

	return msg, nil
}

// Message implements the llm.Message interface
type Message struct {
	Resp   *APIResponse
	Choice *Choice
}

func (m *Message) Role() string {
	return m.Choice.Message.Role
}

func (m *Message) Content() string {
	if m.Choice.Message.Content == nil {
		return ""
	}
	return *m.Choice.Message.Content
}

func (m *Message) ToolCalls() []llm.ToolCall {
	var calls []llm.ToolCall
	for _, call := range m.Choice.Message.ToolCalls {
		calls = append(calls, &ToolCallWrapper{call})
	}
	return calls
}

func (m *Message) ToolResponse() (id string, ok bool) {
	id = m.Choice.Message.ToolCallID
	ok = id != ""
	return
}

func (m *Message) StatUsage() (int, int) {
	return m.Resp.Usage.PromptTokens, m.Resp.Usage.CompletionTokens
}

// ToolCallWrapper implements llm.ToolCall
type ToolCallWrapper struct {
	Call ToolCall
}

func (t *ToolCallWrapper) ID() string {
	return t.Call.ID
}

func (t *ToolCallWrapper) Name() string {
	return t.Call.Function.Name
}

func (t *ToolCallWrapper) Arguments() map[string]any {
	var args map[string]any
	if err := json.Unmarshal([]byte(t.Call.Function.Arguments), &args); err != nil {
		return make(map[string]any) // TODO(xsw): optimize
	}
	return args
}
