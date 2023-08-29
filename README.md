# VALL-E

## About

VALL-E: Re-implementation of VALL-E paper

## Method

- [x] Use [G2P](https://github.com/Kyubyong/g2p/) or [MFA](https://montreal-forced-aligner.readthedocs.io/) to encode text
- [x] Use [EnCodec](https://github.com/facebookresearch/audiocraft) to
tokenise and detokensize audio
- [x] Custom LJSpeech dataloader to include phonemes and EnCodec audio tokens

## Model Parameters

Model from the paper uses a GPT-2 `small` like model with roughly
152M params for the AR, and NAR each?

- `d_model`  := 512 <!-- 1024 -->
- `n_heads`  := 16
- `n_layers` := 12

## TODO

- [x] LJSpeech code
- [x] EnCodec + G2P code (original paper uses Montreal Forced Aligner?)
- [ ] Create model
   - [ ] AR model
   - [ ] NAR model
- [ ] Overfit on one sample from LJSpeech