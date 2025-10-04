from backend.testing.transcript_proccessing.transcript_wav2vec_pipeline import transcribe_single_file

def evaluate(file_path: str):

    print("evaluating file", file_path)

    res_tuple = transcribe_single_file(file_path, "test.json")
    res = res_tuple[0]
    audio = res_tuple[1]


    segments_response = []
    for segment in res["segments"]:
        segments_response.append({
            "text": segment["text"],
            "startTime": {
                "minute": int(segment["start"] / 60),
                "second": segment["start"] % 60
            },
            "endTime": {
                "minute": int(segment["end"] / 60),
                "second": segment["end"] % 60
            },
            "extreme": 0.5
        })

    return {
        "segments": segments_response,
    }, audio

    # TODO: DO SOME PROCESSING HERE

    # segments = whisper(file_path)
    # for s in segments:
    #    analyze(s)
    # result = ....


    return {}
