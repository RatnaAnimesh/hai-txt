from dhai4_hdc.models.level0_broca import HDC_SensoryInterface
broca = HDC_SensoryInterface(dimension=10000)
stream = ["the", "cat", "jumped", "the", "dog", "ran"]
for w in stream:
    broca.encode(w)

the_vec = broca.hd_space.generate_atomic_vector("the")
tiger_vec = broca.hd_space.generate_atomic_vector("tiger")
broca.prev_word_vec = the_vec
broca.determine_role("tiger", tiger_vec)

c1 = broca.word_contexts["cat"]
c2 = broca.word_contexts["tiger"]
print("cat vs tiger:", broca.hd_space.similarity(c1, c2))
