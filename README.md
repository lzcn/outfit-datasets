# Fashion outfit datasets for recommendation

[![Documentation Status](https://readthedocs.org/projects/outfit-datasets/badge/?version=latest)](https://outfit-datasets.readthedocs.io/en/latest/?badge=latest)

This project was created with the intention to provide a unified interface for fashion outfit datasets processing and data loading, but is not fully completed and might contain unfinished parts.

I am fully aware that this project has much room for improvement. Since I am currently moving away my intresting from this topic, I may not able to provide any support or updates. However, this proejct is open for anyone to use, modify, and distribute without any restrictions.

See the [documentation](https://outfit-datasets.readthedocs.io/en/latest/) for more details.

## Installation

```bash
git clone https://github.com/lzcn/outfit-datasets.git
python setup.py install
```

After installation, you can go to each sub-folder to process the dataset:
   - `iqon-3000`
   - `maryland-polyvore`
   - `polyvore-180`
   - `polyvore-outfits`
   - `polyvore-u`
   - `ifashion`
   - `shift15m`


## Citation

If you find this project useful, please consider citing the following papers:

```bibtex
@inproceedings{LearningBinaryCodeLu19,
  title = {Learning Binary Code for Personalized Fashion Recommendation},
  booktitle = {The {{IEEE Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  author = {Lu, Zhi and Hu, Yang and Jiang, Yunchao and Chen, Yan and Zeng, Bing},
  year = {2019},
  pages = {10562--10570},
  doi = {10.1109/CVPR.2019.01081}
}

@inproceedings{PersonalizedOutfitRecommendationLu21,
  title = {Personalized Outfit Recommendation with Learnable Anchors},
  booktitle = {Proceedings of the {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}}},
  author = {Lu, Zhi and Hu, Yang and Chen, Yan and Zeng, Bing},
  year = {2021},
  pages = {12722--12731},
  doi = {10.1109/CVPR46437.2021.01253},
  urldate = {2021-08-11}
}

@article{LearningFashionCompatibilityLu23,
  title = {Learning Fashion Compatibility with Context Conditioning Embedding},
  author = {Lu, Zhi and Hu, Yang and Yu, Cong and Chen, Yan and Zeng, Bing},
  year = {2023},
  journal = {IEEE Transactions on Multimedia},
  pages = {5516--5526},
  issn = {1941-0077},
  doi = {10.1109/TMM.2022.3193560}
}

@article{PersonalizedFashionRecommendationLu23,
  title = {Personalized Fashion Recommendation with Discrete Content-Based Tensor Factorization},
  author = {Lu, Zhi and Hu, Yang and Yu, Cong and Jiang, Yunchao and Chen, Yan and Zeng, Bing},
  year = {2023},
  journal = {IEEE Transactions on Multimedia},
  volume = {25},
  pages = {5053--5064},
  issn = {1941-0077},
  doi = {10.1109/TMM.2022.3186744}
}
```