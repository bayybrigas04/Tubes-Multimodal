SEA-VL Indonesian cultural retrieval dataset — student bundle

2347 images with bilingual captions (Indonesian + English).

Files
-----
  images/          one image per row, named svl_<wikipedia_id>.jpg
  train.csv        training split
  val.csv          validation split
  ATTRIBUTION.md   licence and citation obligations (read this)

Manifest columns
----------------
  image_id     unique identifier, matches the file name in images/
  file         relative image path
  caption_en   English caption
  caption_id   Indonesian caption
  native_lang  source language label
  loc          cultural location(s) of the subject
  category     coarse cultural category (keyword based, may be refined)
  relevance    mean annotator score for cultural relevance (3.0-5.0)
  caption_fit  mean annotator score for caption/image fit (0.67-1.0)
  img_quality  mean annotator score for image quality (0.67-1.0)
  n_annotator  number of independent annotators (2-4)
  width/height stored image dimensions

The test split is deliberately withheld. It is released in the evaluation week.
