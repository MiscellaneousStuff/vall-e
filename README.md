# VALL-E

## About

VALL-E: Re-implementation of VALL-E paper

## Method

- [x] Use [G2P](https://github.com/Kyubyong/g2p/) or [MFA](https://montreal-forced-aligner.readthedocs.io/) to encode text
- [x] Use [EnCodec](https://github.com/facebookresearch/audiocraft) to
tokenise and detokensize audio
- [x] Custom LJSpeech dataloader to include phonemes and EnCodec audio tokens

## LJSpeech

- [ ] Overfit model on one sample from LJSpeech