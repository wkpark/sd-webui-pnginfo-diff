import html
import re
import gradio as gr

from difflib import Differ

from modules import script_callbacks, shared, images, ui_common, deepbooru
from modules.call_queue import wrap_gradio_call

import modules.generation_parameters_copypaste as parameters_copypaste

DEFAULT_NEGATIVE = 'SimpleNegative, EasyNegative, (badhandv4), negative_hand-neg, ng_deepnegative_v1_75t, (worst quality:1.3), (low quality:1), (normal quality:1.4), lowres, skin spots, acnes, skin blemishes, age spot, glan, extra fingers, fewer fingers, strange fingers, bad hand, bad anatomy, fused fingers, missing leg, mutated hand, malformed limbs, missing feet, multiple legs, extra hands, extra foots, lori, '

def interrogator(image):
    if image is None:
        return '', '', '', '', '', ''

    # call interrogate
    prompt = shared.interrogator.interrogate(image.convert("RGB"))

    return prepare_pnginfo(image, prompt)

def interrogate_deepbooru(image):
    if image is None:
        return '', '', '', '', '', ''

    # deepboru
    prompt = deepbooru.model.tag(image)

    return prepare_pnginfo(image, prompt)

def prepare_pnginfo(image, prompt):
    # replace _ to spaces
    prompt = prompt.replace('_', ' ')

    # read geninfo if it available
    geninfo, _ = images.read_info_from_image(image)

    info = ""
    if geninfo is not None:
        res, lastline = parse_prompt(geninfo)
        info = "Prompt by deepbooru/interrogate"
        negative = res["Negative prompt"]
    else:
        clip_skip = shared.opts.CLIP_stop_at_last_layers
        negative = shared.opts.data.get("pnginfo_diff_default_neg_prompt", DEFAULT_NEGATIVE)
        w, h = image.size
        lastline = f"Steps: 20, Sampler: DPM++ 2M Karras, CFG scale:7, Seed: -1, Size: {w}x{h}, Clip skip: {clip_skip}"
        geninfo = prompt + "\n\n" + "Negative prompt: " + negative + "\n" + lastline
        info = "Prompt by deepbooru/interrogate, default negative prompt used, image size detected"

    return '', geninfo, prompt, negative, lastline, info

def run_pnginfo(image):
    if image is None:
        return '', '', '', '', '', ''

    geninfo, items = images.read_info_from_image(image)

    if geninfo is not None:
        res, lastline = parse_prompt(geninfo)
    else:
        res, lastline = {}, ''

    info = ''
    for key, text in items.items():
        info += f"""
<h3>{html.escape(str(key))}</h3>
<p>
{plaintext_to_html(str(text))}
</p>
""".strip()+"\n"

    if len(res) == 0 and len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"
        return '', geninfo, '', '', lastline, info
    if "Prompt" not in res.keys():
        res["Prompt"] = ''
    if "Negative prompt" not in res.keys():
        res["Negative prompt"] = ''

    return '', geninfo, res["Prompt"], res["Negative prompt"], lastline, info

def plaintext_to_html(text):
    text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
    return text

re_param_code = r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)

def parse_prompt(x: str):
    # simple version of modules/generation_parameters_copypaste.py
    """parses prompt parameters string, the one you see in text field under the picture in UI:
```
girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
```

    returns a dict with field values and remains
    """

    res = {}

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    if shared.opts.infotext_styles != "Ignore":
        found_styles, prompt, negative_prompt = shared.prompt_styles.extract_styles_from_prompt(prompt, negative_prompt)

        if shared.opts.infotext_styles == "Apply":
            res["Styles array"] = found_styles
        elif shared.opts.infotext_styles == "Apply if any" and found_styles:
            res["Styles array"] = found_styles

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    return res, lastline

def parse_remain_paramters(lastline: str):
    res = {}
    for k, v in re_param.findall(lastline):
        try:
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)

            m = re_imagesize.match(v)
            if m is not None:
                res[f"{k}-1"] = m.group(1)
                res[f"{k}-2"] = m.group(2)
            else:
                res[k] = v
        except Exception:
            print(f"Error parsing \"{k}: {v}\"")

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        res["Prompt"] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    ##restore_old_hires_fix_params(res)

    # Missing RNG means the default was set, which is GPU RNG
    if "RNG" not in res:
        res["RNG"] = "GPU"

    if "Schedule type" not in res:
        res["Schedule type"] = "Automatic"

    if "Schedule max sigma" not in res:
        res["Schedule max sigma"] = 0

    if "Schedule min sigma" not in res:
        res["Schedule min sigma"] = 0

    if "Schedule rho" not in res:
        res["Schedule rho"] = 0

    return res

def diff_texts(direct, text1, text2):
    d = Differ()
    if direct == "A-B":
        return [
            (token[2:], token[0] if token[0] != " " else None)
            for token in d.compare(text1, text2)
        ]
    else:
        return [
            (token[2:], token[0] if token[0] != " " else None)
            for token in d.compare(text2, text1)
        ]

def add_tab():
    with gr.Blocks(analytics_enabled=False) as pnginfo_diff:
        with gr.Row().style(equal_height=False):
            with gr.Column(scale=4, variant='compact'):
                image1 = gr.Image(elem_id="pnginfo_image1", label="Source", source="upload", interactive=True, type="pil")

            with gr.Column(scale=1, elem_classes="interrogate-col"):
                interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")
                pnginfo = gr.Button('Get\nPNG Info', elem_id="pnginfo")

            with gr.Column(scale=6, variant='compact'):
                html1 = gr.HTML()
                prompt1 = gr.Textbox(label="Prompt", elem_id="prompt1", show_label=False, interactive=False, placeholder="Prompt", lines=3, elem_classes=["prompt"])
                negative1 = gr.Textbox(label="Negative prompt", elem_id="neg_prompt1", interactive=False, show_label=False, placeholder="Negative prompt", lines=2, elem_classes=["prompt"])
                extra1 = gr.Textbox(label="Extra", elem_id="extra11", show_label=False, interactive=False, placeholder="Seed, Model...", lines=1, elem_classes=["prompt"])
                generation_info1 = gr.Textbox(visible=False, elem_id="pnginfo_generation_info2")
                html1a = gr.HTML()
                with gr.Row():
                    buttons1 = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])

                for tabname, button in buttons1.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=generation_info1, source_image_component=image1,
                    ))

        with gr.Row().style(equal_height=False):
            with gr.Column(scale=4, variant='compact'):
                image2 = gr.Image(elem_id="pnginfo_image2", label="Source", source="upload", interactive=True, type="pil")

            with gr.Column(scale=1, elem_classes="interrogate-col"):
                interrogate2 = gr.Button('Interrogate\nCLIP', elem_id="interrogate2")
                deepbooru2 = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru2")
                pnginfo2 = gr.Button('Get\nPNG Info', elem_id="pnginfo")

            with gr.Column(scale=6, variant='compact'):
                html2 = gr.HTML()
                prompt2 = gr.Textbox(label="Prompt", elem_id="prompt2", show_label=False, interactive=False, placeholder="Prompt", lines=3, elem_classes=["prompt"])
                negative2 = gr.Textbox(label="Negative prompt", elem_id="neg_prompt2", interactive=False, show_label=False, placeholder="Negative prompt", lines=2, elem_classes=["prompt"])
                extra2 = gr.Textbox(label="Extra", elem_id="extra11", show_label=False, interactive=False, placeholder="Seed, Model...", lines=1, elem_classes=["prompt"])
                generation_info2 = gr.Textbox(visible=False, elem_id="pnginfo_generation_info2")
                html2a = gr.HTML()
                with gr.Row():
                    buttons2 = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])

                for tabname, button in buttons2.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=generation_info2, source_image_component=image2,
                    ))

        with gr.Column():
            with gr.Row():
                highlighted = gr.HighlightedText(
                    label="Diff",
                    combine_adjacent=True,
                    show_legend=True,
                ).style(color_map={"-": "red", "+": "green"})

            with gr.Row():
                direct = gr.Radio(["A-B", "B-A"], label="Direction", show_label=False, value="A-B")
                prompt_diff = gr.Button(value="Prompt diff")
                neg_prompt_diff = gr.Button(value="Negative prompt diff")
                extra_diff = gr.Button(value="Extra info diff")

        prompt_diff.click(
            fn=diff_texts,
            inputs=[direct, prompt1, prompt2],
            outputs=[highlighted]
        )

        neg_prompt_diff.click(
            fn=diff_texts,
            inputs=[direct, negative1, negative2],
            outputs=[highlighted]
        )

        extra_diff.click(
            fn=diff_texts,
            inputs=[direct, extra1, extra2],
            outputs=[highlighted]
        )

        image1.change(
            fn=run_pnginfo,
            inputs=[image1],
            outputs=[html1, generation_info1, prompt1, negative1, extra1, html1a],
        )

        image2.change(
            fn=run_pnginfo,
            inputs=[image2],
            outputs=[html2, generation_info2, prompt2, negative2, extra2, html2a],
        )

        pnginfo.click(
            fn=run_pnginfo,
            inputs=[image1],
            outputs=[html1, generation_info1, prompt1, negative1, extra1, html1a],
        )

        pnginfo2.click(
            fn=run_pnginfo,
            inputs=[image2],
            outputs=[html2, generation_info2, prompt2, negative2, extra2, html2a],
        )

        interrogate.click(
            fn=interrogator,
            inputs=[image1],
            outputs=[html1, generation_info1, prompt1, negative1, extra1, html1a],
        )

        deepbooru.click(
            fn=interrogate_deepbooru,
            inputs=[image1],
            outputs=[html1, generation_info1, prompt1, negative1, extra1, html1a],
        )

        interrogate2.click(
            fn=interrogator,
            inputs=[image2],
            outputs=[html2, generation_info2, prompt2, negative2, extra2, html2a],
        )

        deepbooru2.click(
            fn=interrogate_deepbooru,
            inputs=[image2],
            outputs=[html2, generation_info2, prompt2, negative2, extra2, html2a],
        )
    return [(pnginfo_diff, "PNG Info Diff", "pnginfo_diff")]

def on_ui_settings():
    shared.opts.add_option("pnginfo_diff_default_neg_prompt", shared.OptionInfo(DEFAULT_NEGATIVE, "Default Negative prompt", section=("pnginfo_diff", "PNGInfo Diff")))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(add_tab)
