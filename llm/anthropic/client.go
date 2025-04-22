package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type Client struct {
	apiKey  string
	client  *http.Client
	baseURL string
}

func NewClient(apiKey string, baseURL string, client *http.Client) *Client {
	return new(Client).Init(apiKey, baseURL, client)
}

func (c *Client) Init(apiKey string, baseURL string, client *http.Client) *Client {
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	} else if !strings.HasSuffix(baseURL, "/v1") {
		baseURL = strings.TrimSuffix(baseURL, "/") + "/v1"
	}
	if client == nil {
		client = http.DefaultClient
	}
	c.apiKey = apiKey
	c.baseURL = baseURL
	c.client = client
	return c
}

func (c *Client) CreateMessage(ctx context.Context, req CreateRequest) (*APIMessage, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	url := c.baseURL + "/messages"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Api-Key", c.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
			return nil, fmt.Errorf("error response with status %d", resp.StatusCode)
		}

		if errResp.Error.Type == "overloaded_error" {
			return nil, fmt.Errorf("overloaded_error: %s", errResp.Error.Message)
		}

		return nil, fmt.Errorf("%s: %s", errResp.Error.Type, errResp.Error.Message)
	}

	var message APIMessage
	if err := json.NewDecoder(resp.Body).Decode(&message); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &message, nil
}
