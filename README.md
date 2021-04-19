# Epoching_StyleGan2
A small repo containing code from a variety of sources (cited in the README.md) for the purpose of latent space exploration!

My fastpages blog post/notebook: [Latent-Space-Exploration-with-StyleGAN2](https://amarsaini.github.io/Epoching-Blog/jupyter/2020/08/10/Latent-Space-Exploration-with-StyleGAN2.html)

## Steps

`pip install requests Pillow tqdm cmake dlib`

## Tests

### cmd line interface

Get help
`python cmd_latent.py -h`

Make a gif with the age direction.
`python cmd_latent.py --task gif --feature age`

Get a jpg with the age direction.
`python cmd_latent.py --task direction --feature age --amount 2 --latent-file 'inputs/dalton_latent_code.pkl'`

## Sources

- [align_face.py](https://gist.github.com/lzhbrian/bde87ab23b499dd02ba4f588258f57d5)
- [stylegan2](https://github.com/NVlabs/stylegan2)
- [stylegan2directions](https://twitter.com/robertluxemburg/status/1207087801344372736)
