# seedpod 中文手册

seedpod 是一个统一的、与模型厂商解耦的 Go 语言 LLM 客户端。它把 OpenAI、Anthropic、Gemini、Ollama 以及你自己的模型服务封装到同一套 API 后面，支持多模态消息、流式输出、推理/思考块、运行时 Provider 注册和请求费用估算。

> 本项目的设计目标与 llmhub 类似：用一套简洁 API 接入多个大模型厂商，并允许业务代码在不同 Provider 之间平滑切换。当前仓库模块名为 `github.com/teoclub/seedpod`。

## 为什么使用 seedpod

- **一套 API，多个厂商**：业务代码面向 `seedpod.Client` 编写，Provider 可按需替换。
- **多模态消息**：一次请求中可以混合文本和图片。
- **流式响应**：通过 Go channel 消费增量内容。
- **推理块保留**：部分模型会返回 reasoning/thinking 内容，seedpod 会归一化保存。
- **可扩展注册表**：可以在运行时注册第一方或第三方 Provider。
- **函数式选项**：用 `provider.WithModel`、`provider.WithBaseURL`、`provider.WithCost` 等方式配置请求。

## 安装

```bash
go get github.com/teoclub/seedpod
```

按需导入 Provider。Provider 使用空白导入自注册：

```go
import (
	_ "github.com/teoclub/seedpod/provider/openai"
	_ "github.com/teoclub/seedpod/provider/anthropic"
	_ "github.com/teoclub/seedpod/provider/gemini"
	_ "github.com/teoclub/seedpod/provider/ollama"
)
```

## 快速开始

```go
package main

import (
	"context"
	"fmt"

	"github.com/teoclub/seedpod"
	"github.com/teoclub/seedpod/provider"
	"github.com/teoclub/seedpod/schema"

	_ "github.com/teoclub/seedpod/provider/openai"
)

func main() {
	client, err := seedpod.New("openai", "sk-YOUR-KEY",
		provider.WithModel("gpt-4o-mini"),
	)
	if err != nil {
		panic(err)
	}

	prompt := []*schema.Message{
		schema.NewSystemMessage(schema.Text("你是一个幽默但可靠的助手。")),
		schema.NewUserMessage(schema.Text("用五个字解释量子力学。")),
	}

	resp, err := client.Generate(context.Background(), prompt)
	if err != nil {
		panic(err)
	}

	fmt.Println(resp.Text())
}
```

## 核心概念

### Client

`seedpod.Client` 是主入口：

- `Generate(ctx, prompt, opts...)`：一次性生成完整响应。
- `Stream(ctx, prompt, opts...)`：返回 `<-chan schema.StreamChunk`，用于流式消费。
- `ProviderName()`：返回底层 Provider 名称。

可以通过 `seedpod.New(providerName, apiKey, opts...)` 从注册表创建客户端，也可以用 `seedpod.Wrap(p, opts...)` 包装已构造好的自定义 Provider。

### Message 与 ContentPart

消息类型位于 `schema` 包：

- `schema.NewSystemMessage(...)`
- `schema.NewUserMessage(...)`
- `schema.NewAssistantMessage(...)`
- `schema.NewToolMessage(...)`

内容块支持：

- `schema.Text("...")`
- `schema.Image("https://example.com/a.png")`
- `schema.Image("data:image/png;base64,...")`
- `schema.Reasoning("...")`

`schema.Response` 会把结果统一放入 `Content []schema.ContentPart`。常见文本场景可以直接调用：

```go
fmt.Println(resp.Text())
fmt.Println(resp.ReasoningText())
```

## 流式响应

```go
stream, err := client.Stream(ctx, prompt)
if err != nil {
	log.Fatal(err)
}

for chunk := range stream {
	if chunk.Err != nil {
		log.Printf("stream error: %v", chunk.Err)
		break
	}
	if chunk.ReasoningDelta != "" {
		log.Printf("reasoning delta: %s", chunk.ReasoningDelta)
	}
	fmt.Print(chunk.Delta)
	if chunk.Done {
		break
	}
}
```

`schema.StreamChunk` 字段说明：

- `Delta`：普通文本增量。
- `ReasoningDelta`：推理/思考内容增量。
- `Done`：流结束。
- `Err`：流中错误。

## 视觉与多模态输入

可以在用户消息中同时传入文本和图片：

```go
prompt := []*schema.Message{
	schema.NewUserMessage(
		schema.Text("请描述这张图。"),
		schema.Image("https://example.com/diagram.png"),
	),
}

resp, err := client.Generate(ctx, prompt)
if err != nil {
	log.Fatal(err)
}

fmt.Println(resp.Text())
```

图片可以是 URL，也可以是 data URL。CLI 会把本地图片文件自动转换成 data URL。

不同 Provider 对图片能力不同：

| Provider | 输入图片 |
| --- | --- |
| OpenAI | 支持 `image_url` |
| Anthropic | 支持 URL 与 base64 data URL |
| Gemini | 支持 data URL；非 data URL 会作为 file URI 传入 |
| Ollama | 当前实现仅支持纯文本消息 |

## 推理 / Thinking 块

部分模型会把推理或 thinking 内容作为独立字段返回。seedpod 会把它们归一化成 `*schema.ReasoningContent`。

```go
resp, err := client.Generate(ctx, prompt)
if err != nil {
	log.Fatal(err)
}

fmt.Println("最终答案:", resp.Text())
fmt.Println("推理内容:", resp.ReasoningText())

for _, part := range resp.Content {
	if r, ok := part.(*schema.ReasoningContent); ok {
		fmt.Println("reasoning block:", r.Text)
	}
}
```

流式场景中，推理增量位于 `StreamChunk.ReasoningDelta`。

## Provider 注册表

Provider 通过全局注册表创建。内置 Provider 会在包被导入时自动注册。

```go
func init() {
	seedpod.MustRegisterProvider("my-llm",
		func(apiKey string, opts ...provider.Option) (provider.Provider, error) {
			return newMyClient(apiKey, opts...)
		},
	)
}
```

消费者只需要：

```go
client, err := seedpod.New("my-llm", "token")
```

查看已注册 Provider：

```go
fmt.Println(seedpod.RegisteredProviders())
```

## 函数式选项

常用选项位于 `provider` 包：

- `provider.WithAPIKey(key)`：设置凭证。也可以直接把 key 传给 `seedpod.New`。
- `provider.WithModel(model)`：选择模型。
- `provider.WithBaseURL(url)`：指向代理、自托管网关或本地服务。
- `provider.WithTemperature(value)`：设置采样温度，默认 `0.7`。
- `provider.WithMaxTokens(n)`：设置输出 token 硬上限。
- `provider.WithHTTPClient(client)`：替换 HTTP 客户端。
- `provider.WithHeader(k, v)`：追加自定义请求头。
- `provider.WithWebSearch(true)`：启用联网搜索/grounding。目前 Gemini 会使用 `google_search` 工具。
- `provider.WithResponseModalities("IMAGE")`：控制输出模态，主要用于 Gemini 图片生成。
- `provider.WithCost(input, output)`：配置每百万输入/输出 token 的美元价格，用于费用估算。

建议在普通业务代码中谨慎使用 `WithMaxTokens`。过低的上限容易导致输出被截断；没有明确硬限制时，优先使用 Provider 默认值。

## 内置 Provider

| Provider | 默认模型 | 默认 Base URL | 说明 |
| --- | --- | --- | --- |
| OpenAI | `gpt-4o-mini` | `https://api.openai.com/v1` | Chat Completions，多模态输入，SSE 流式输出 |
| Anthropic | `claude-3-haiku-20240307` | `https://api.anthropic.com` | Messages API，支持 system 独立字段和流式输出 |
| Gemini | `gemini-1.5-flash` | `https://generativelanguage.googleapis.com` | 支持多模态、web search、图片输出模态 |
| Ollama | `llama3` | `http://localhost:11434` | 本地 `/api/chat`，支持流式输出 |

### OpenAI 细节

自定义 `BaseURL` 时，OpenAI Provider 会确保地址以 `/v1` 结尾。下面两种写法等价：

```go
provider.WithBaseURL("https://api.openai.com")
provider.WithBaseURL("https://api.openai.com/v1")
```

如果模型设置为 `"default"`，Provider 初始化时会请求 `/v1/models`，并选择返回列表中的第一个模型。这对 Ollama、vLLM、LocalAI 等 OpenAI 兼容服务很有用：

```go
client, err := seedpod.New("openai", "key",
	provider.WithBaseURL("http://localhost:11434"),
	provider.WithModel("default"),
)
```

### Anthropic 细节

Anthropic Provider 会把 `system` 角色消息提取到 Messages API 的 `system` 字段中。当前实现要求 system 消息只包含文本内容。

默认 `MaxTokens` 为 `1024`，因为 Anthropic Messages API 要求请求中包含 `max_tokens`。

### Gemini 细节

Gemini Provider 支持 `WithWebSearch(true)`：

```go
client, err := seedpod.New("gemini", apiKey,
	provider.WithModel("gemini-2.5-flash"),
	provider.WithWebSearch(true),
)
```

对于图片生成模型，可以通过 `WithResponseModalities` 请求图片输出：

```go
client, err := seedpod.New("gemini", apiKey,
	provider.WithModel("gemini-2.5-flash-image"),
	provider.WithResponseModalities("IMAGE"),
)
```

如果希望允许文本和图片混合输出：

```go
provider.WithResponseModalities("TEXT", "IMAGE")
```

生成图片会作为 `*schema.ImageContent` 返回，通常是 data URL：

```go
for _, part := range resp.Content {
	if img, ok := part.(*schema.ImageContent); ok {
		preview := img.URL
		if len(preview) > 64 {
			preview = preview[:64] + "..."
		}
		fmt.Println("image:", preview)
	}
}
```

### Ollama 细节

Ollama Provider 默认连接本机：

```go
client, err := seedpod.New("ollama", "",
	provider.WithModel("qwen3:32b"),
	provider.WithBaseURL("http://localhost:11434"),
)
```

如果你使用远程 Ollama 或兼容代理，可以替换 `BaseURL`，也可以通过 `WithHeader` 或 `WithAPIKey` 添加认证信息。

## 费用估算

seedpod 会根据 Provider 返回的 token 使用量和你配置的单价计算费用。单价单位是「每 100 万 token 的美元价格」。

```go
client, err := seedpod.New("openai", apiKey,
	provider.WithModel("gpt-4o"),
	provider.WithCost(2.50, 10.00),
)
if err != nil {
	log.Fatal(err)
}

resp, err := client.Generate(ctx, prompt)
if err != nil {
	log.Fatal(err)
}

fmt.Printf("Tokens: %d in, %d out\n",
	resp.Usage.PromptTokens,
	resp.Usage.CompletionTokens,
)
fmt.Printf("Cost: $%.6f\n", resp.Usage.Cost)
```

计算公式：

```text
Cost = PromptTokens * InputRate / 1,000,000
     + CompletionTokens * OutputRate / 1,000,000
```

未配置价格时，`resp.Usage.Cost` 为 `0`。

也可以在单次请求中覆盖价格：

```go
resp, err := client.Generate(ctx, prompt,
	provider.WithCost(0.15, 0.60),
)
```

## CLI 测试工具

仓库包含一个命令行工具，方便端到端测试 Provider：

```bash
go run ./examples/cli [options]
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-provider` | Provider 名称：`openai`、`anthropic`、`gemini`、`ollama`，必填 |
| `-model` | 模型名称 |
| `-api-key` | API Key；也可使用环境变量 |
| `-base-url` | 自定义 Provider Base URL |
| `-prompt` | 直接传入提示词 |
| `-prompt-file` | 从文件读取提示词 |
| `-images` | 逗号分隔的图片路径或 URL |
| `-stream` | 启用流式模式 |
| `-temperature` | 采样温度，默认 `0.7` |
| `-max-tokens` | 输出 token 硬上限；不需要时留空 |
| `-input-cost` | 每 100 万输入 token 的美元价格 |
| `-output-cost` | 每 100 万输出 token 的美元价格 |
| `-timeout` | 请求超时时间，如 `30s`、`2m`、`10m` |

### CLI 示例

使用 Ollama：

```bash
go run ./examples/cli \
  -provider ollama \
  -model qwen3:32b \
  -base-url http://localhost:11434 \
  -prompt "为什么天空是蓝色的？"
```

使用 Gemini：

```bash
go run ./examples/cli \
  -provider gemini \
  -model gemini-2.5-flash \
  -api-key YOUR_GEMINI_KEY \
  -prompt "请用简单语言解释量子纠缠。"
```

Gemini 视觉输入：

```bash
go run ./examples/cli \
  -provider gemini \
  -model gemini-2.5-flash \
  -api-key YOUR_GEMINI_KEY \
  -prompt "详细描述这张图片。" \
  -images cat.jpg
```

OpenAI 流式输出：

```bash
go run ./examples/cli \
  -provider openai \
  -model gpt-4o \
  -api-key YOUR_OPENAI_KEY \
  -prompt "写一首关于编程的俳句。" \
  -stream
```

使用环境变量：

```bash
export OPENAI_API_KEY=sk-...
go run ./examples/cli -provider openai -model gpt-4o -prompt "你好！"
```

费用估算：

```bash
go run ./examples/cli \
  -provider openai \
  -model gpt-4o \
  -input-cost 2.50 \
  -output-cost 10.00 \
  -prompt "解释 Go interface。"
```

## 多 Provider 路由

如果业务需要运行时选择厂商，可以为每个 Provider 创建一个 `Client`：

```go
openaiClient := seedpod.MustNew("openai", os.Getenv("OPENAI_API_KEY"),
	provider.WithModel("gpt-4o"),
)

claudeClient := seedpod.MustNew("anthropic", os.Getenv("ANTHROPIC_API_KEY"),
	provider.WithModel("claude-3-haiku-20240307"),
)

func answer(ctx context.Context, prompt []*schema.Message, vendor string) (*schema.Response, error) {
	switch vendor {
	case "anthropic":
		return claudeClient.Generate(ctx, prompt)
	default:
		return openaiClient.Generate(ctx, prompt)
	}
}
```

## 测试

运行全部测试：

```bash
go test ./...
```

## 参考源码

https://github.com/smhanov/llmhub
