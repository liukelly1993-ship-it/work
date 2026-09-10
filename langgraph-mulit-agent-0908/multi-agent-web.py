import gradio as gr
import random
from typing import Any, cast
from Director import graph


def process_input(text: str):
    """Generator: 同时订阅 writer 日志 + LLM token 流,每条事件 yield 一次"""
    # 类型标注:config 是 LangGraph 的 RunnableConfig(dict 形态)
    config: Any = {"configurable": {"thread_id": random.randint(1, 1000)}}
    log_lines: list[str] = []
    accumulated = ""

    # ⭐ cast:让 Pylance 不再抱怨 dict literal 不匹配 RouterState(运行时完全 OK)
    initial_state = cast(Any, {"query": text})

    # stream_mode=["custom","messages"] 时 IDE 推断不到精确 tuple 类型,加 type: ignore
    for mode, event in graph.stream(  # type: ignore[call-overload]
        initial_state,
        config,
        stream_mode=["custom", "messages"],
    ):
        if mode == "custom":
            # custom 模式:writer() 写出来的原始数据(可能是 dict / str)
            if isinstance(event, dict):
                for k, v in event.items():
                    log_lines.append(f"• [{k}] {v}")
            else:
                log_lines.append(f"• {event}")

        elif mode == "messages":
            # messages 模式:(BaseMessageChunk, RunMetadata) 元组
            msg_chunk, metadata = event
            node = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
            if node == "synthesize_results_node" and msg_chunk.content:
                if isinstance(msg_chunk.content, str):
                    accumulated += msg_chunk.content
                else:
                    accumulated += "".join(
                        c.get("text", "") for c in msg_chunk.content
                        if isinstance(c, dict)
                    )

        yield "\n".join(log_lines) + (
            "\n\n📝 回复:\n" + accumulated if accumulated else ""
        )


# 自定义 CSS 样式
custom_css = """
#main-container {
    max-width: 1200px;
    margin: 0 auto;
}
#header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
#header h1 {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}
#header p {
    font-size: 1.1rem;
    opacity: 0.9;
}
.feature-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.example-btn {
    margin: 0.25rem;
}
#output-box {
    min-height: 200px;
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
}
.footer {
    text-align: center;
    padding: 1rem;
    color: #666;
    font-size: 0.9rem;
    margin-top: 2rem;
}
"""
# 创建界面
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    # 标题区域
    with gr.Column(elem_id="header"):
        gr.Markdown("# 🤖 LangGraph 多智能体助手")
        gr.Markdown("基于 LangGraph 的智能对话系统 | 路线规划 · 对联创作 · 笑话娱乐")

    with gr.Row(elem_id="main-container"):
        # 左侧输入区域
        with gr.Column(scale=1):
            gr.Markdown("### 💬 输入您的问题")

            inputs_text = gr.Textbox(
                label="",
                placeholder="输入您的问题...",
                lines=3,
                value="讲一个郭德纲的笑话"
            )

            with gr.Row():
                btn_start = gr.Button("🚀 开始提问", variant="primary",
                                      size="lg")
                btn_clear = gr.ClearButton([inputs_text], value="🗑 清空",
                                           size="lg")

            gr.Markdown("---")
            gr.Markdown("### 📝 示例问题")

            # 示例问题区域
            with gr.Column():
                gr.Markdown("**🗺 路线规划**", elem_classes="feature-card")
                ex1 = gr.Button("规划从北京天安门到故宫的路线", size="sm",
                                elem_classes="example-btn")
                ex2 = gr.Button("给我规划一条从上海人民广场到虹桥火车站的驾车路线",
                                size="sm", elem_classes="example-btn")

                gr.Markdown("**✍ 对联创作**", elem_classes="feature-card")
                ex3 = gr.Button("帮我对个对联，上联是：瑞雪兆丰年", size="sm",
                                elem_classes="example-btn")
                ex4 = gr.Button("帮我对个对联，上联是：金榜题名时", size="sm",
                                elem_classes="example-btn")
                gr.Markdown("**😄 笑话娱乐**", elem_classes="feature-card")
                ex5 = gr.Button("讲一个郭德纲的笑话", size="sm",
                                elem_classes="example-btn")
                ex6 = gr.Button("给我讲个关于程序员的笑话", size="sm",
                                elem_classes="example-btn")

        # 右侧输出区域
        with gr.Column(scale=1):
            gr.Markdown("### 💡 智能体回复")
            output_text = gr.Textbox(
                label="",
                lines=15,
                elem_id="output-box"
                # show_copy_button=True
            )

            # 状态指示器
            status = gr.Markdown("", visible=False)

    # 页脚
    with gr.Row():
        gr.Markdown(
            """
            <div class="footer">
                <p>🔧 Powered by LangGraph & LangChain | 支持多种智能体协作 | Made 
with ❤</p>
            </div>
            """,
            elem_classes="footer"
        )


    # 事件绑定
    def show_loading():
        return gr.update(visible=True, value="⏳ 智能体正在处理中...")


    def hide_loading():
        return gr.update(visible=False)


    # 主按钮点击事件
    btn_start.click(
        fn=process_input,
        inputs=[inputs_text],
        outputs=[output_text]
    )

    # 示例按钮点击事件
    ex1.click(lambda: "规划从北京天安门到故宫的路线", None, inputs_text)
    ex2.click(lambda: "给我规划一条从上海人民广场到虹桥火车站的驾车路线", None,
              inputs_text)
    ex3.click(lambda: "帮我对个对联，上联是：瑞雪兆丰年", None, inputs_text)
    ex4.click(lambda: "帮我对个对联，上联是：金榜题名时", None, inputs_text)
    ex5.click(lambda: "讲一个郭德纲的笑话", None, inputs_text)
    ex6.click(lambda: "给我讲个关于程序员的笑话", None, inputs_text)
# 启动应用
if __name__ == "__main__":
    demo.queue(max_size=10).launch(  # ⭐ queue() 必须加,才能真正流式输出
        server_name="0.0.0.0",  # 允许外部访问
       server_port=7860,        # 指定端口
        share=False,             # 是否创建公共链接
        show_error=True,          # 显示错误信息
    )
