# Retina_Anomaly_Detection

Within OCT images there are some illnesses that can be detected using anomaly detection techniques. As a first approach, a normal retina layer functions are modelled by getting their respective polynomial functions and then, those very far from the normal distribution, are detected as anomalies.

To execute the algorithm, just execute read_image.py file and, within that file change imshow to True and imwrite flag to True if you want to see and to write the results to a disk.

For now, the function normalizes the image, computes the histogram, automatically computes a mask and compute sobel within the mask to have the row data to estimate the main layers of the retina automatically. It is left to do the conversion to a continuous function to estimate the polinomail function relative to each layer and then properly model each layer for each cut of a OCT image.
