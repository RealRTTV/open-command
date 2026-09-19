import json
import subprocess
from typing import List

import numpy as np
import tensorrt as trt
import torch
import time
import queue
import threading

clip_url = "/home/riley/.cache/statcast-subsidiary/sporty-video/00002752-4764-3cd1-b6fd-d60093629cf7.mp4"

def decode_thread(url: str, decode_queue: queue.Queue):
    ffmpeg = subprocess.Popen(f"ffmpeg -hwaccel cuda -i {url} -f rawvideo -pix_fmt rgb24 -loglevel error -", stdout=subprocess.PIPE, shell=True)
    assert ffmpeg.stdout is not None, "FFMPEG proces has no pipe stdout"
    try:
        while len(buf := ffmpeg.stdout.read(1280 * 720 * 3)) >= 1280 * 720 * 3:
            decode_queue.put(buf)
        decode_queue.put(None)
    finally:
        ffmpeg.stdout.close()
        ffmpeg.kill()
        ffmpeg.wait()

decode_queue = queue.Queue(maxsize=32)
thread = threading.Thread(target=decode_thread, args=(clip_url, decode_queue), daemon=True)

runtime = trt.Runtime(logger=trt.Logger(min_severity=trt.Logger.VERBOSE))
with open("/home/riley/src/open-command/models/ball.engine", "rb") as f:
    length = int.from_bytes(f.read(4), byteorder="little")
    engine_metadata = json.loads(f.read(length))
    engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(f.read())
context: trt.IExecutionContext = engine.create_execution_context()
tensor_names: dict[trt.TensorIOMode, str] = dict(map(lambda name: (engine.get_tensor_mode(name), name), map(engine.get_tensor_name, range(engine.num_io_tensors))))

input_tensor_name = tensor_names[trt.TensorIOMode.INPUT]
output_tensor_name = tensor_names[trt.TensorIOMode.OUTPUT]

input_buffer = torch.empty(tuple(engine.get_tensor_shape(input_tensor_name)), dtype=torch.float16, device="cuda")
output_buffer = torch.empty(tuple(engine.get_tensor_shape(output_tensor_name)), dtype=torch.float16, device="cuda")

context.set_tensor_address(input_tensor_name, input_buffer.data_ptr())
context.set_tensor_address(output_tensor_name, output_buffer.data_ptr())

frame_batch_uint8 = torch.empty((16, 704, 704, 3), dtype=torch.uint8, device="cuda")
def read_batch_of_frames() -> int:
    bufs = []
    for _ in range(16):
        buf = decode_queue.get()
        if buf is None:
            break
        bufs.append(np.frombuffer(buf, dtype=np.uint8).reshape(720, 1280, 3)[8:712, 288:992])
    n = len(bufs)
    if n == 0:
        return 0

    for i in range(n):
        frame_batch_uint8[i].copy_(torch.from_numpy(bufs[i]))

    input_tensor_data = frame_batch_uint8[:n].permute(0, 3, 1, 2).to(torch.float16).div_(255)
    input_buffer[:n].copy_(input_tensor_data)
    return n

def execute_inference():
    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream)
    torch.cuda.current_stream().synchronize()

start = time.time()
thread.start()
mp4_frame = 0
while True:
    n = read_batch_of_frames()
    if n == 0:
        break

    results = execute_inference()
    # for result in results[:n]:
    #     boxes = result.boxes
    #
    #     if len(boxes) > 0:
    #         conf = float(boxes[0].conf)
    #         cx = float(boxes.xywh[0, 0])
    #         cy = float(boxes.xywh[0, 1])
    #     else:
    #         conf = 0.0
    #         cx = np.nan
    #         cy = np.nan
    #     # print(f"frame: {mp4_frame}, cx: {cx}, cy: {cy}, conf: {conf}")
    #     mp4_frame += 1

print(f"took {time.time() - start}s")
