import gradio as gr
import whisper
import gc
import argparse

# available arguments to launch the script
parser = argparse.ArgumentParser(description="Launch Gradio app with custom settings.")
parser.add_argument("--port", type=int, default=7860, help="The port to use for the web server.")
parser.add_argument("--listen", type=str, default=None, help="The IP address to listen on.")
parser.add_argument("--share", action="store_true", help="Whether to create a public link for sharing.")

args = parser.parse_args()

languages = {"":"... (detect language)", "af_za": "Afrikaans", "am_et": "Amharic", "ar_eg": "Arabic", "as_in": "Assamese", "az_az": "Azerbaijani", "be_by": "Belarusian", "bg_bg": "Bulgarian", "bn_in": "Bengali", "bs_ba": "Bosnian", "ca_es": "Catalan", "cmn_hans_cn": "Chinese", "cs_cz": "Czech", "cy_gb": "Welsh", "da_dk": "Danish", "de_de": "German", "el_gr": "Greek", "en_us": "English", "es_419": "Spanish", "et_ee": "Estonian", "fa_ir": "Persian", "fi_fi": "Finnish", "fil_ph": "Tagalog", "fr_fr": "French", "gl_es": "Galician", "gu_in": "Gujarati", "ha_ng": "Hausa", "he_il": "Hebrew", "hi_in": "Hindi", "hr_hr": "Croatian", "hu_hu": "Hungarian", "hy_am": "Armenian", "id_id": "Indonesian", "is_is": "Icelandic", "it_it": "Italian", "ja_jp": "Japanese", "jv_id": "Javanese", "ka_ge": "Georgian", "kk_kz": "Kazakh", "km_kh": "Khmer", "kn_in": "Kannada", "ko_kr": "Korean", "lb_lu": "Luxembourgish", "ln_cd": "Lingala", "lo_la": "Lao", "lt_lt": "Lithuanian", "lv_lv": "Latvian", "mi_nz": "Maori", "mk_mk": "Macedonian", "ml_in": "Malayalam", "mn_mn": "Mongolian", "mr_in": "Marathi", "ms_my": "Malay", "mt_mt": "Maltese", "my_mm": "Myanmar", "nb_no": "Norwegian", "ne_np": "Nepali", "nl_nl": "Dutch", "oc_fr": "Occitan", "pa_in": "Punjabi", "pl_pl": "Polish", "ps_af": "Pashto", "pt_br": "Portuguese", "ro_ro": "Romanian", "ru_ru": "Russian", "sd_in": "Sindhi", "sk_sk": "Slovak", "sl_si": "Slovenian", "sn_zw": "Shona", "so_so": "Somali", "sr_rs": "Serbian", "sv_se": "Swedish", "sw_ke": "Swahili", "ta_in": "Tamil", "te_in": "Telugu", "tg_tj": "Tajik", "th_th": "Thai", "tr_tr": "Turkish", "uk_ua": "Ukrainian", "ur_pk": "Urdu", "uz_uz": "Uzbek", "vi_vn": "Vietnamese", "yo_ng": "Yoruba"}
models = {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v2", "large-v3"}

# Track the currently loaded model name and model object
current_model_name = None
current_model = None


def load_model(model_name):
    global current_model, current_model_name
    # If the selected model is already loaded, do nothing
    if model_name == current_model_name:
        return current_model
    else:
        # Unload the current model by dereferencing and calling the garbage collector
        del current_model
        gc.collect()
        # Load the new model
        current_model = whisper.load_model(model_name)
        current_model_name = model_name
        print(f"Model loaded: {model_name} to {current_model.device}")
        return current_model

def format_timestamps_srt(timestamps):
    srt_format = ""
    for i, segment in enumerate(timestamps):
        # Convert start and end times from seconds to the SRT time format
        start_time = format_time_srt(segment['start'])
        end_time = format_time_srt(segment['end'])
        # Each segment of subtitles in SRT format
        srt_format += f"{i+1}\n{start_time} --> {end_time}\n{segment['text']}\n\n"
    return srt_format

def format_time_srt(time_in_seconds):
    # Convert the time in seconds to hours, minutes, seconds, and milliseconds
    hours, remainder = divmod(time_in_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    # Format the time according to SRT specifications: hours:minutes:seconds,milliseconds
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"


def inference(model_name, audio, language_code, translate):
    model = load_model(model_name)
    audio = whisper.load_audio(audio)
    options = dict(beam_size=5, best_of=5)

    if language_code:
      options = dict(language=language_code, **options)

    transcribe_options = dict(task="transcribe", **options)
    translate_options = dict(task="translate", **options)

    if translate:
        result = model.transcribe(audio, **translate_options)
    else:
        result = model.transcribe(audio, **transcribe_options)

    transcription = result["text"]
    timestamps = result["segments"]
    timestamp_str = format_timestamps_srt(timestamps)
#    print(transcription)
    return [transcription, timestamp_str]


block = gr.Blocks()

with block:
    with gr.Row(equal_height=True):
      with gr.Column(scale=1, min_width=300):
        model_dropdown = gr.Dropdown(label="Select Whisper Model", choices=list(models), value="large-v3")
        language_dropdown = gr.Dropdown(label="Select Language", choices=list(languages.values()), value="English", info="Select the language of your original audio file. The quality of transcription is muc highier when you explicitely pass the language into the model. If you don't know the language, you can leave it empty, but don't use the LARGE-v3 model is this case")
        translate_checkbox = gr.Checkbox(label="Translate to English", value=False, info="Whisper can translate any transcription to English and only to English")
      with gr.Column(scale=2):
        audio = gr.Audio(label="Input Audio", show_label=False, sources=["upload", "microphone"], type="filepath")
    with gr.Row():
      btn = gr.Button("Transcribe")
    with gr.Row():
      text = gr.Textbox(label="Transcription", show_label=True, elem_id="transcribe-textarea")
    with gr.Row():
      timestamps = gr.Textbox(label="Srt format transcription", info="You can save this file as an .srt file and use in any video player", show_label=True, elem_id="timestamped-textarea")

    btn.click(inference, inputs=[model_dropdown, audio, language_dropdown, translate_checkbox], outputs=[text, timestamps])

block.launch(
    server_port=args.port,
    server_name=args.listen,
    share=args.share
)
