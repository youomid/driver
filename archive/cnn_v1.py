shape_counts = defaultdict(int)
for i, cat in enumerate(glob.glob('cats/*')[:1000]):
    if i%100==0:
        print(i)
    img_shape = pixels_from_path(cat).shape #loads image as np matrix and checks shape.
    shape_counts[str(img_shape)]= shape_counts[str(img_shape)]+ 1
    
shape_items = list(shape_counts.items())
shape_items.sort(key = lambda x: x[1])
shape_items.reverse()