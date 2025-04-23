package anthropic

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

func NewProvider(apiKey string, baseURL string, client *http.Client, model string) *Provider {
	if model == "" {
		model = "claude-3-5-sonnet-20240620"
	}
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

	anthropicMessages := make([]MessageParam, 0, len(messages))

	for _, msg := range messages {
		log.Debug("converting message",
			"role", msg.Role(),
			"content", msg.Content(),
			"is_tool_response", llm.IsToolResponse(msg))

		content := []ContentBlock{}

		// Add regular text content if present
		if textContent := strings.TrimSpace(msg.Content()); textContent != "" {
			content = append(content, ContentBlock{
				Type: "text",
				Text: textContent,
			})
		}

		// Add tool calls if present
		for _, call := range msg.ToolCalls() {
			input, _ := json.Marshal(call.Arguments())
			content = append(content, ContentBlock{
				Type:  "tool_use",
				ID:    call.ID(),
				Name:  call.Name(),
				Input: input,
			})
		}

		// Handle tool responses
		if toolCallID, ok := msg.ToolResponse(); ok {
			log.Debug("processing tool response",
				"tool_call_id", toolCallID,
				"raw_message", msg)

			if historyMsg, ok := msg.(*history.HistoryMessage); ok {
				for _, block := range historyMsg.AContent {
					if block.Type == "tool_result" {
						content = append(content, ContentBlock{
							Type:      "tool_result",
							ToolUseID: block.ToolUseID,
							Content:   block.Content,
						})
					}
				}
			} else {
				// Always include tool response content
				content = append(content, ContentBlock{
					Type:      "tool_result",
					ToolUseID: toolCallID,
					Content:   msg.Content(),
				})
			}
		}

		// Always append the message, even if content is empty
		// This maintains conversation flow
		anthropicMessages = append(anthropicMessages, MessageParam{
			Role:    msg.Role(),
			Content: content,
		})
	}

	// Add the new prompt if provided
	if prompt != "" {
		anthropicMessages = append(anthropicMessages, MessageParam{
			Role: "user",
			Content: []ContentBlock{{
				Type: "text",
				Text: prompt,
			}},
		})
	}

	// Convert tools to Anthropic format
	anthropicTools := make([]Tool, len(tools))
	for i, tool := range tools {
		anthropicTools[i] = Tool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: InputSchema{
				Type:       tool.InputSchema.Type,
				Properties: tool.InputSchema.Properties,
				Required:   tool.InputSchema.Required,
			},
		}
	}

	log.Debug("sending messages to Anthropic",
		"messages", anthropicMessages,
		"num_tools", len(tools))

	// Make the API call
	resp, err := p.client.SendMessage(ctx, CreateRequest{
		Model:     p.model,
		Messages:  anthropicMessages,
		MaxTokens: 4096,
		Tools:     anthropicTools,
	})
	if err != nil {
		return nil, err
	}

	return &Message{Msg: *resp}, nil
}

func (p *Provider) SupportsTools() bool {
	return true
}

func (p *Provider) Name() string {
	return "anthropic"
}

func (p *Provider) CreateToolResponse(toolCallID string, content any) (llm.Message, error) {
	log.Debug("creating tool response",
		"tool_call_id", toolCallID,
		"content_type", reflect.TypeOf(content),
		"content", content)

	var contentStr string
	var structuredContent any = content

	// TODO(xsw): check contentStr
	// Convert content to string if needed
	switch v := content.(type) {
	case string:
		contentStr = v
	case []byte:
		contentStr = string(v)
	default:
		// For structured content, create JSON representation
		if jsonBytes, err := json.Marshal(content); err == nil {
			contentStr = string(jsonBytes)
		} else {
			contentStr = fmt.Sprintf("%v", content)
		}
	}

	msg := &Message{
		Msg: APIMessage{
			Role: "tool",
			Content: []ContentBlock{{
				Type:      "tool_result",
				ToolUseID: toolCallID,
				Content:   structuredContent, // Original structure
				Text:      contentStr,        // String representation
			}},
		},
	}

	return msg, nil
}
