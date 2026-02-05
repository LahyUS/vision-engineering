import numpy as np
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        # Initialize the next unique object ID to be assigned
        self.next_object_id = 0
        
        # Ordered dictionary to store object ID and its centroid
        self.objects = OrderedDict()
        
        # Ordered dictionary to store number of consecutive frames
        # an object ID has been marked as "disappeared"
        self.disappeared = OrderedDict()
        
        # Max number of frames an object is allowed to be marked as
        # "disappeared" until we deregister it
        self.max_disappeared = max_disappeared
        
        # Max distance between centroids to associate an object -- 
        # if distance is larger than this, they are different objects
        self.max_distance = max_distance

    def register(self, centroid):
        # Register a new object
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # Deregister an object ID
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # Check to see if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # Loop over any existing tracked objects and mark them as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # Initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        # Loop over the bounding box rectangles
        for (i, (start_x, start_y, width, height)) in enumerate(rects):
            # Use the bounding box coordinates to derive the centroid
            # stored as (delta_x, delta_y) relative to top-left is not enough, 
            # we need center point: cX = x + w/2, cY = y + h/2
            c_x = int(start_x + width / 2.0)
            c_y = int(start_y + height / 2.0)
            input_centroids[i] = (c_x, c_y)

        # If we are currently not tracking any objects, take the input centroids
        # and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])

        # Otherwise, match input centroids to existing object centroids
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute the distance between each pair of object centroids and
            # input centroids
            # Manual Euclidean distance calculation using broadcasting
            # D[i, j] is dist between object_centroid[i] and input_centroid[j]
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # In order to perform this matching we must (1) find the smallest
            # value in each row and then (2) sort the row indexes based on their
            # minimum values so that the row with the smallest value is at the
            # *front* of the index list
            rows = D.min(axis=1).argsort()

            # Next, we perform a similar process on the columns by finding the
            # smallest value in each column and then sorting using the
            # previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # In order to keep track of which of the rows and column indexes
            # we have already examined, update two sets
            used_rows = set()
            used_cols = set()

            # Loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # If we have already examined either the row or column, ignore it
                if row in used_rows or col in used_cols:
                    continue

                # If the distance between centroids is greater than the max
                # distance, do not associate the two
                if D[row, col] > self.max_distance:
                    continue

                # Otherwise, grab the object ID for the current row, set its
                # new centroid, and reset the disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # Indicate that we have examined each of the row and column indexes
                used_rows.add(row)
                used_cols.add(col)

            # Compute both the row and column indexes we have NOT yet examined
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # In the event that the number of object centroids is equal or
            # greater than the number of input centroids
            # we need to check and see if some of these objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # Otherwise, we need to register each new input centroid as a trackable object
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        # Return the set of trackable objects
        return self.objects
