import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import cv2
import datascience as ds
import time

def capture_check(model):
  """Function that takes a photo and returns a table utilizing a YOLOV5 MODEL.
  The returned table has columns class, confidence, and name. It is sorted descending
  by confidence."""
  capture = cv2.VideoCapture(1)
  result, image = capture.read()
  look = model(image)
  table = ds.Table().from_df(look.pandas().xyxy[0])
  table = table.sort("confidence", descending=True).select("class", "confidence", "name")
  return table

def one_shot(adjacency, target):
  """Runs capture_check for ADJACENCY number of times. If TARGET is found
  within class 0 of the resultant table all ADJACENCY times, return True.
  Else return False."""
  adjacent_match = 0
  print(f"DEBUG: Capturing {adjacency} images...")
  while adjacent_match < adjacency:
    print(f"DEBUG: {adjacent_match+1}")
    if target in capture_check(model).where("class", 0).column("name"):
      adjacent_match += 1
    else:
       return False
  return True

def poller(freq, adjacency, target, match, fail, model):
  """Function that runs capture_check at a specified FREQ (seconds).
  Looks for a TARGET in the resulting table. If it finds a match for ADJACENCY
  subsequent calls, run the MATCH function, else run the FAIL function."""
  while True:
    print(f"\nDEBUG: Current Time = {time.time()}")
    if one_shot(adjacency, target):
      print(f"DEBUG: {target} found in all {adjacency} images")
      match
    else:
      print(f"DEBUG: {target} NOT found in all {adjacency} images")
      fail
    print(f"DEBUG: sleeping for {freq} seconds...")
    time.sleep(freq)


model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

poller(5, 3, "person", print(""), print(""), model)
