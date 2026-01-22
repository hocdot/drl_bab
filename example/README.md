# TODO

- [x] add example net and spec
- [x] add linear bound propagation
- [x] calculate polarity score from intermediate bound by deep poly
- [ ] implement branch and bound for deeppoly, bounding just using naive output bound.
    todo this, I want W_L, W_U, b_L, b_U to be bounding of layer i compared to i-1,
    not fully back-substituted to x
- [ ] choose weight to map intermediate bounds, bias, mask to guided score
- [ ] tuning weight and bias
