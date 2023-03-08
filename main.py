import gradio as gr
import whisper

model = whisper.load_model("small")

def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    transcription = result.text
    print(transcription)
    return transcription

block = gr.Blocks()

with block:
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                audio = gr.Audio(
                    label="Input Audio",
                    show_label=False,
                    source="microphone",
                    type="filepath"
                )
                btn = gr.Button("Transcribe")
        text = gr.Textbox(show_label=False, elem_id="result-textarea")
        btn.click(inference, inputs=audio, outputs=text)

block.launch()
