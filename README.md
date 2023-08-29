# VALL-E

## About

VALL-E: Re-implementation of VALL-E paper

## Method

- [ ] Use [G2P](https://github.com/Kyubyong/g2p/) or [MFA](https://montreal-forced-aligner.readthedocs.io/) to encode text
- [ ] Use [EnCodec](https://github.com/facebookresearch/audiocraft) to
tokenise and detokensize audio
- [ ] Custom LJSpeech dataloader to include phonemes and EnCodec audio tokens