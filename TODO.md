is the sequence returned from decode.beam_search the same as the sequence returned in the training.validate_one_step?

make sure that with the new rotary embedding the result is still the same, NOT THE SAME, DIFFERENT OUTPUT

need to rewrite the rotary embedding into a function or split class and function
i need to be able to wrap the function with @torch.fx.wrap so that it is not traced even though cos / sin are detached