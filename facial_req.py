#! /usr/bin/python

# import the necessary packages
from imutils.video import FileVideoStream
import imutils
import face_recognition
import pickle
import time
import os, glob
import multiprocessing
from multiprocessing import Process, Pipe, current_process
from multiprocessing.connection import wait

#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "/home/pi/facial_recognition/encodings.pickle"
#Path to video files
video_path = "/home/pi/motion/snapshots/"

def process_frame(frame):
  try:
    frame = imutils.resize(frame, width=500)
    # Detect the fce boxes
    boxes = face_recognition.face_locations(frame)
    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(frame, boxes)

    # loop over the facial embeddings
    for encoding in encodings:
      # attempt to match each face in the input image to our known
      # encodings
      matches = face_recognition.compare_faces(data["encodings"], encoding)

      # check to see if we have found a match
      if True in matches:
        # find the indexes of all matched faces then initialize a
        # dictionary to count the total number of times each face
        # was matched
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
          name = data["names"][i]
          counts[name] = counts.get(name, 0) + 1

        # determine the recognized face with the largest number
        # of votes (note: in the event of an unlikely tie Python
        # will select first entry in the dictionary)
        name = max(counts, key=counts.get)

        #If someone in your dataset is identified, print their name on the screen
        #print("Found", name)
        return name
  except:
    print("exception")
  return ""

def worker(q):
  while True:
    try:
      frame = q.recv()
      if frame is not None:
        #print(multiprocessing.current_process().name)
        q.send(process_frame(frame))
      else:
        print("read null frame")
        q.send("")
    except:
      print("oops, some exception")
      q.send("")

# loop over frames from the video file stream until all done
# returns name of person recognised
def recognise(video_file):
  start_time = time.time()
  # initialize the video file
  print("processing", video_file)
  try:
    vs = FileVideoStream(video_file).start()
  except:
    print("can't open video file")
    return ""
  done = False
  name = ""
  result = ""
  qs = []
  ps = []
  # create processes and send first frame to each
  for i in range(3):
    qp, qc = Pipe()
    qs.append(qp)
    p = Process(target=worker, args=(qc,))
    ps.append(p)
    p.start()
    qc.close()
    try:
      frame = vs.read()
      if frame is not None:
        qp.send(frame)
      else:
        done = True
        break
    except:
      print("can't read frame")
      done = True
  # wait for one or more replies to come in
  while not done:
    worker_queues = wait(qs)
    while (len(worker_queues) > 0) and not done:
      result = worker_queues[0].recv()
      done = len(result) > 0
      if done:
        name = result
        print(name)
      try:
        frame = vs.read()
      except:
        print("can't read frame")
        frame = None
      if frame is not None:
        worker_queues[0].send(frame)
        worker_queues.pop(0)
      else:
        done = True
  # now kill all the processes we created
  for p in ps:
    p.terminate()
  # do a bit of cleanup
  vs.stop()

  if not done:
    print("recognition failed")

  end_time = time.time()
  print("elapsed time = {:.2f}".format(end_time - start_time))
  os.remove(video_file)
  return name

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

def distribute():
  PROCESSES = 3
  print('Creating pool with %d processes\n' % PROCESSES)

  with multiprocessing.Pool(PROCESSES) as pool:
    TASKS = glob.glob(os.path.join(video_path, '*.mkv'))
    imap_it = pool.imap(recognise, TASKS)
    for r in imap_it:
      print('\t', r)
    print()

if __name__ == '__main__':
  multiprocessing.freeze_support()
  #distribute()
  for filename in glob.glob(os.path.join(video_path, '*.mkv')):
    recognise(filename)
