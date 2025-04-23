package llm

import (
	"context"
)

// Message represents a message in the conversation.
type Message interface {
	// Role returns the role of the message sender (e.g., "user",
	// "assistant", "system").
	Role() string

	// Content returns the text content of the message.
	Content() string

	// ToolCalls returns any tool calls made in this message.
	ToolCalls() []ToolCall

	// ToolResponse returns (toolCallID, true) if this message is a
	// response from a tool.
	ToolResponse() (toolCallID string, is bool)

	// StatUsage returns token usage statistics if available
	StatUsage() (input int, output int)
}

// IsToolResponse returns if this message is a response from a tool.
func IsToolResponse(msg Message) bool {
	_, is := msg.ToolResponse()
	return is
}

// ToolCall represents a tool invocation.
type ToolCall interface {
	// Name returns the tool's name.
	Name() string

	// Arguments returns the arguments passed to the tool.
	Arguments() map[string]any

	// ID returns the unique identifier for this tool call.
	ID() string
}

// Tool represents a tool definition.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	InputSchema Schema `json:"input_schema"`
}

// Schema defines the input parameters for a tool.
type Schema struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Required   []string       `json:"required"`
}

// Provider defines the interface for LLM providers.
type Provider interface {
	// SendMessage sends a message to the LLM and returns the response.
	SendMessage(ctx context.Context, prompt string, messages []Message, tools []Tool) (Message, error)

	// CreateToolResponse creates a message representing a tool response.
	CreateToolResponse(toolCallID string, content any) (Message, error)

	// SupportsTools returns whether this provider supports tool/function calling.
	SupportsTools() bool

	// Name returns the provider's name.
	Name() string
}
