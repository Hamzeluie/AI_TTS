import time
import torch
from vexu.AI_TTS.src.fish_text2speech import FishEngine
from RealtimeTTS import TextToAudioStream
from multiprocessing import freeze_support


def dummy_generator():
    text = "مرحبا يا جماعة، كيفكم؟ اليوم حاب أحكي عن شغلة بنمر فيها كلنا، يمكن ما بنعطيها اهتمام، بس إلها تأثير كبير علينا، الضغط اليومي. يعني من أول ما نصحى، وإحنا برتم سريع، شغل، جامعة، مشاوير، التزامات، والزلمة ما بلحق حتى يشرب قهوته ع رواق. كل يوم نفس الروتين، وكأننا داخلين سباق، وكل ما نقول خلص، هاليوم بمر، بنلاقي حالنا بدوامة تانية! طب وإمتى آخر مرة فعلياً قعدت مع حالك؟ لا تليفون، لا تليفونات، لا مشاغل، بس لحظة هدوء؟ صرنا ننسى نعيش اللحظة، نضحك من قلبنا، نستمتع بشغلة بسيطة زي كاسة شاي، أو قعدة عالبلكونة وقت المغرب. أنا بقول، لازم نروق شوي، نخفف السرعة، ونفكر بنفسنا شوي، لأنه صحتك النفسية مش إشي ثانوي، هاد أهم إشي. خدلك لحظة كل يوم، احكي مع حالك، رتب أفكارك، وافهم شو بتحتاج، الدنيا مش مستاهلة كل هالركض، وإذا ما ريحت حالك، محدا رح يجي يريحك. وبس، هاي كانت"
    # text = "Okay, the user said No! after I asked for their name. Since they're refusing to provide the information, I should move on to the next step. According to the guidelines, if the caller avoids providing information after one prompt, I should move on. So I'll skip asking for their name and proceed to ask for the reason for the call. I need to keep the response under onewords and in Arabic since the conversation is in Arabic. I'll confirm their request and ask for the reason. Let's make sure to stay professional and concise."
    # for substr in text.split():
    #     yield substr + " "
    print("Starting dummy generator...")
    yield text  # Yield the entire text at once for simplicity

# Initialize the global variable before use
in_chunk_ts = None

def on_audio_chunk_callback(chunk):
    global in_chunk_ts
    if in_chunk_ts is None:
        in_chunk_ts = time.time()
    else:
        elapsed = time.time() - in_chunk_ts
        print(f"Chunk received after {elapsed:.2f} seconds")
        in_chunk_ts = time.time()
    # print(f"Chunk received, len: {len(chunk)}")

def main():
    engine = FishEngine(
        checkpoint_dir="/home/ubuntu/ai_service/checkpoints/openaudio-s1-mini",
        output_dir="/home/ubuntu/ai_service/outputs", 
        voices_path=None,
        reference_prompt_tokens="/home/ubuntu/ai_service/ref_kelly.npy",  # or "my_voice.wav"
        reference_prompt_text="""topic but I think honestly I didn't there was no thought process to this album for me really it was all just I mean thought process in the sense of I'm going through something and trying to figure out where I'm at and what I'm feeling and what I'm going to do that but like as far as like oh this is a song that I'm going to put on a record like that wasn't it um it was really just a lot of it like before you got here I was listening to with your people and I was like, man, this is like, I was very angry. Yeah. And foundation being hurt. So yeah, so it's really just, yeah, I didn't really think about it. And I do like the idea of taking a quirky pop happy sound melodically and like the sound of the""",
        device="cuda",
        precision=torch.bfloat16#"half",
    )
    # Try direct synthesis first without the stream
    # print("Testing direct synthesis...")
    # success = engine.synthesize("This is a test.")
    # print(f"Direct synthesis result: {success}")
    ts = time.time()
    print("Starting TextToAudioStream...")
    stream = TextToAudioStream(engine, muted=True).feed(dummy_generator())
    stream.play_async(
        tokenizer=None,
        language="en",
        on_audio_chunk=on_audio_chunk_callback,
        muted=True,
        output_wavfile="/home/ubuntu/ai_service/outputs/test_output_ar_4.wav",
    )

    # while stream.is_playing():
    #     time.sleep(0.1)
    print(f"Stream finished in {time.time() - ts:.2f} seconds")
    # engine.shutdown()

if __name__ == '__main__':
    freeze_support()  # Required for multiprocessing on Windows
    main()
    # Initialize the engine
    
    # # Get available voices
    # voices = engine.get_voices()
    # print(f"Available voices: {voices}")

    # # Set a different voice
    # engine.set_voice("another_voice.wav")

    # # Synthesize text
    # success = engine.synthesize("Hello, this is a test of the Fish TTS engine!")

    # # The audio will be available in engine.queue
    # while not engine.queue.empty():
    #     audio_chunk = engine.queue.get()
        # Process audio_chunk (bytes)

    # Shutdown when done
