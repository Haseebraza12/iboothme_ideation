import gradio as gr
import re
import random
import json
from dotenv import load_dotenv
import os
from openai import OpenAI

# 🔐 Load API Key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 1) Load your product catalog once at startup
def load_product_catalog(path="products.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expecting products_desc to be a dict: { "iboothme X": "…", … }
    return data["products_desc"]

PRODUCT_CATALOG = load_product_catalog()

# 2) Helper to pull full descriptions for a list of product names
def get_product_descriptions(product_names: list[str]) -> str:
    entries = []
    for name in product_names:
        desc = PRODUCT_CATALOG.get(name)
        if desc:
            entries.append(f"**{name}**\n{desc}")
    return "\n\n".join(entries)

# Keyword extraction from the paragraph
def extract_keywords(paragraph: str) -> list[str]:
    prompt = f"""
You are an expert in experiential event planning.

Extract 5-10 short, specific, and thematic keywords or concepts from the event description below. These will be used to inspire immersive, tech-powered event ideas.

Each keyword should be 2-4 words long and describe a concrete idea or theme (e.g., "photo booths", "smart vending", "interactive storytelling").

Event Description:
\"{paragraph}\"

Return the keywords as a comma-separated list.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    raw = response.choices[0].message.content
    return [kw.strip().lower() for kw in re.split(r'[,\n]', raw) if kw.strip()]

# (Optional) Keyword extraction from titles/links
def extract_keywords_from_title_and_link(title: str, link: str) -> list[str]:
    prompt = f"""
You are an expert in event innovation.

Given the title and link below, extract 3-5 short, specific, and meaningful keywords or themes (2-4 words each) that describe what the page is about.

Title: {title}
Link: {link}

Return the keywords as a comma-separated list.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=150
    )
    raw = response.choices[0].message.content
    return [kw.strip().lower() for kw in re.split(r'[,\n]', raw) if kw.strip()]

# (Optional) Web search for inspiration
def search_similar_events_and_products_openai(keywords: list[str]) -> list[tuple[str,str]]:
    input_text = f"Generate 10 useful URLs for experiential event ideas or iboothme.com inspiration related to the keywords: {', '.join(keywords)}"
    try:
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=input_text
        )
        content = response.output_text
        results = []
        for line in content.split("\n"):
            if "http" in line:
                parts = line.split(" - ", 1)
                if len(parts) == 2:
                    results.append((parts[0].strip(), parts[1].strip()))
                else:
                    url = line.strip()
                    results.append((url, url))
        return results[:10]
    except Exception as e:
        print("Search failed:", e)
        return []

# Core idea generation, enriched with random product descriptions
def generate_event_ideas(
    paragraph: str,
    product_info: str,
    search_links: list[tuple[str,str]],
    all_keywords: list[str],
    idea_count: int
) -> str:
    search_summary = "\n".join([f"- {title}: {url}" for title, url in search_links])
    include_games = any("game" in kw for kw in all_keywords)
    game_instruction = "Include at least two game-related ideas (e.g., quiz game, vending challenge)." if include_games else ""

    prompt = f"""
You are an expert event strategist for iboothme, a company offering creative experiences like AI photo booths, smart vending machines, audio booths, personalization stations, and immersive visual storytelling.

Below are full descriptions of three randomly selected iboothme products, to keep all ideas on-brand:
{product_info}

Based on the event description below, generate {idea_count} unique and diverse iboothme-powered event ideas.
Make sure that the syntax of the brand is always "iboothme" (all lowercase).

**Event Description:**
{paragraph}

**Inspiration from Related Ideas:**
{search_summary}

💡 **Your Task:**
Create ideas that are immersive, memorable, and creatively use iboothme's photo, video, and audio-based technologies. Do not use AR, VR, projection mapping, or other tech-heavy elements.

You must include:
- At least two game-related ideas
- Studio Ghibli-inspired visuals in one idea
- Personalized giveaways (e.g., custom t-shirts, stickers, Labibu dolls)
{game_instruction}

❗ Important:
- Avoid AR, VR, holograms, or projection domes
- Do not repeat photo‑booth formats
- Every idea should have a creative title
- Each idea should be described in a paragraph
- Immediately after, write a second paragraph describing the user journey flow

Return **only** the final ideas in markdown format.
"""
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.95,
        max_tokens=1500
    )
    return resp.choices[0].message.content

# Main orchestration
def main_workflow(paragraph: str) -> str:
    print("main_workflow called with paragraph:", paragraph)
    if not paragraph.strip():
        print("No paragraph provided.")
        return "❌ Please enter an event description."

    # 1. Randomly pick 3 products from your catalog
    try:
        gadget_names = random.sample(list(PRODUCT_CATALOG.keys()), k=4)
        print("Randomly selected products:", gadget_names)
        product_info = get_product_descriptions(gadget_names)
    except Exception as e:
        print("Error selecting products:", e)
        return f"❌ Error selecting products: {e}"

    # 2. Gather keywords + optional web search
    try:
        base_kw = extract_keywords(paragraph)
        print("Extracted base keywords:", base_kw)
        links = search_similar_events_and_products_openai(base_kw)
        print("Found links:", links)
        link_kw = []
        for t, u in links:
            kws = extract_keywords_from_title_and_link(t, u)
            print(f"Extracted keywords from link ({t}, {u}):", kws)
            link_kw.extend(kws)
        all_kw = sorted(set(base_kw + link_kw))
        print("All keywords:", all_kw)
    except Exception as e:
        print("Error in keyword extraction or web search:", e)
        return f"❌ Error in keyword extraction or web search: {e}"

    # 3. Generate ideas
    try:
        idea_count = random.choice([5,6,7,8])
        print("Idea count:", idea_count)
        ideas_md = generate_event_ideas(paragraph, product_info, links, all_kw, idea_count)
        print("Generated ideas markdown.")
    except Exception as e:
        print("Error generating ideas:", e)
        return f"❌ Error generating ideas: {e}"

    # 4. Summarize top keywords
    summaries = []
    for kw in all_kw[:10]:
        try:
            print(f"Summarizing keyword: {kw}")
            r = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Give a short one-line event idea description using the keyword: {kw}"}],
                temperature=0.6,
                max_tokens=60
            )
            desc = r.choices[0].message.content.strip()
            summaries.append(f"- **{kw.title()}**: {desc}")
        except Exception as e:
            print(f"Error summarizing keyword {kw}:", e)
            summaries.append(f"- **{kw.title()}**")

    summary_md = "\n".join(summaries)
    print("Returning final markdown output.")
    return f"""
🌐 **Relevant Keywords Summary:**  
{summary_md}

{ideas_md}
"""
# 8) Styling and Gradio interface
custom_theme = gr.themes.Base(
    primary_hue="purple",
    secondary_hue="purple",
    neutral_hue="gray"
).set(
    body_background_fill="white",
    block_background_fill="white",
    block_border_width="2px",
    block_border_color="#a18cd1",
    button_primary_background_fill="linear-gradient(90deg, #a18cd1 0%, #fbc2eb 100%)",
    button_primary_text_color="white",
    input_background_fill="white",
    input_border_color="#a18cd1"
)

custom_css = """
#iboothme-heading {
    font-weight: 900 !important;
    font-size: 2.5rem !important;
    background: linear-gradient(90deg, #a18cd1 0%, #fbc2eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: black;
    margin-bottom: 0.5em;
    text-align: center;
    letter-spacing: 1px;
}
#desc-subheading {
    text-align: center;
    font-size: 1.15rem;
    font-weight: 500;
    color: #6d4fa7;
    margin-bottom: 2em;
}
.gradio-container { min-height: 100vh; background: white !important; }
.gr-box, .gr-input, .gr-button, .gr-markdown, .gr-textbox, .gr-column, .gr-row {
    border-radius: 18px !important;
}
#event-desc-box, #output-box {
    border: 2px solid #a18cd1 !important;
    box-shadow: 0 4px 24px 0 rgba(161,140,209,0.10) !important;
    background: white !important;
}
#generate-btn {
    font-weight: bold;
    font-size: 1.1rem;
    background: linear-gradient(90deg, #a18cd1 0%, #fbc2eb 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px 0 rgba(161,140,209,0.10) !important;
    margin-top: 1.5em;
}
"""

with gr.Blocks(theme=custom_theme, css=custom_css, title="iboothme Event Ideation App") as demo:
    gr.Markdown(
        "<div id='iboothme-heading'>🎉 <b>iboothme Event Idea Generator</b></div>"
        "<div id='desc-subheading'>Describe your event goal and receive interactive, tech‑powered ideas!</div>"
    )
    with gr.Row():
        with gr.Column(scale=2):
            paragraph = gr.Textbox(
                label="📝 Describe Your Event (e.g. Women’s Day, Product Launch)",
                lines=4,
                elem_id="event-desc-box"
            )
        with gr.Column(scale=1, min_width=220):
            submit_btn = gr.Button("🚀 Generate Event Concepts", elem_id="generate-btn")
    output = gr.Markdown(elem_id="output-box")

    submit_btn.click(
        fn=main_workflow,
        inputs=[paragraph],
        outputs=output,
        show_progress=True
    )

demo.launch(inline=False, share=True)