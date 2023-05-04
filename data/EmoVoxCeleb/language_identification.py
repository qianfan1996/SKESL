import os
import azure.cognitiveservices.speech as speechsdk


def get_one_language(filename, speech_key, service_region):
    # Creates an AutoDetectSourceLanguageConfig, which defines a number of possible spoken languages
    auto_detect_source_language_config = \
    speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["de-DE", "zh-CN", "en-US"])

    # Creates a SpeechConfig from your speech key and region
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Sets the Priority (optional, defaults to 'Latency'). Either 'Latency' or 'Accuracy' is accepted.
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_SingleLanguageIdPriority, value='Latency')

    audio_config = speechsdk.audio.AudioConfig(filename=filename)
    # Creates a source language recognizer using a file as audio input, also specify the speech language
    source_language_recognizer = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config)

    # Starts speech language detection, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. It returns the detection text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot detection like command or query.
    # For long-running multi-utterance detection, use start_continuous_recognition() instead.
    result = source_language_recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("RECOGNIZED: {}".format(result))
        detected_src_lang = result.properties[
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
        print("Detected Language: {}".format(detected_src_lang))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Language Detection canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

def get_all_audio_embedding(path):
    num = 0
    for dir in sorted(os.listdir(path)):
        for audio_name in sorted(os.listdir(os.path.join(path, dir))):
            if not audio_name.startswith('.'):
                xxx
                num += 1
    return num


if __name__ == "__main__":
    filepath = "/data/remote_corpus/corpus_211/VoxCeleb2/dev/wav/id00012/2DLq_Kkc1r8/00016.wav"
    speech_key = "99b0c9115c214865ab01d2f0e236cb77"
    service_region = "eastasia"
    get_one_language(filepath, speech_key, service_region)

