from search.rrf import rrf_fuse


class TestRRFFuse:
    def test_single_list_preserves_order(self):
        assert rrf_fuse([["a", "b", "c"]]) == ["a", "b", "c"]

    def test_empty_input(self):
        assert rrf_fuse([]) == []

    def test_document_in_both_lists_outranks_single_list_hit(self):
        # "b" appears in both lists (ranked lower each time) but should still
        # outrank "a", which only appears once at rank 1.
        fused = rrf_fuse([["a", "b"], ["c", "b"]])
        assert fused[0] == "b"

    def test_disjoint_lists_interleave_by_rank(self):
        fused = rrf_fuse([["a", "b"], ["c", "d"]])
        # top-ranked items from each list score equally and both outrank
        # the second-ranked items.
        assert set(fused[:2]) == {"a", "c"}
        assert set(fused[2:]) == {"b", "d"}

    def test_smaller_k_amplifies_rank_differences(self):
        lists = [["a", "b", "c"]]
        tight = dict(zip(lists[0], range(3)))
        # scores should be strictly decreasing regardless of k
        for k in (1, 60):
            fused = rrf_fuse(lists, k=k)
            assert fused == ["a", "b", "c"]
