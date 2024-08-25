# orchard_bouman
Python implementation of unweighted Orchard Bouman clustering

### Requirements:
- NumPy \
Developed in Python 3.10, but anything over Python 3.7 should work

### Installation:
```
pip install orchard-bouman
```

### Usage:
Once your image is created as a NumPy array, simply use:
- `ob = OrchardBouman(image, k)` \
where `image` is the image you want clustered in RGB format and `k` \
is the number of times to split nodes. The final number of nodes = 2^k.
- This will return the split node objects within a list accessible by `ob.nodes`
- A clustered color image can be generated by `ob.construct_image()`

### Considerations:
This package does not handle image reading and saving to keep it versatile.\
Because all images are treated as a NumPy array, it is recommended to\
read in images using OpenCV, as they are natively NumPy arrays.