package google

import (
	"fmt"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"github.com/goplus/xgowiz/llm"
)

type ToolCall struct {
	genai.FunctionCall

	toolCallID int
}

func (t *ToolCall) Name() string {
	return t.FunctionCall.Name
}

func (t *ToolCall) Arguments() map[string]any {
	return t.Args
}

func (t *ToolCall) ID() string {
	return fmt.Sprintf("Tool<%d>", t.toolCallID)
}

type Message struct {
	*genai.Candidate

	toolCallID int
}

func (m *Message) Role() string {
	return m.Candidate.Content.Role
}

func (m *Message) Content() string {
	var sb strings.Builder
	for _, part := range m.Candidate.Content.Parts {
		if text, ok := part.(genai.Text); ok {
			sb.WriteString(string(text))
		}
	}
	return sb.String()
}

func (m *Message) ToolCalls() []llm.ToolCall {
	var calls []llm.ToolCall
	for i, call := range m.Candidate.FunctionCalls() {
		calls = append(calls, &ToolCall{call, m.toolCallID + i})
	}
	return calls
}

func (m *Message) ToolResponse() (toolCallID string, is bool) {
	for _, part := range m.Candidate.Content.Parts {
		if _, ok := part.(*genai.FunctionResponse); ok {
			return fmt.Sprintf("Tool<%d>", m.toolCallID), true
		}
	}
	return
}

func (m *Message) StatUsage() (input int, output int) {
	return 0, 0
}
