from rag_eval.retrievers import bm25_tokenize, reciprocal_rank_fusion


def test_bm25_tokenize_keeps_identifiers_whole_and_unfolds_ligatures():
    assert bm25_tokenize("What does GV-1.3-007 recommend?") == ["what", "does", "gv-1.3-007", "recommend"]
    assert bm25_tokenize("ID.AM-04: Inventories") == ["id.am-04", "inventories"]
    assert bm25_tokenize("proﬁle") == ["profile"]


def test_bm25_tokenize_drops_trailing_punctuation():
    assert bm25_tokenize("criticality. Next") == ["criticality", "next"]


def test_rrf_rewards_items_ranked_well_by_both_lists():
    dense, sparse = [1, 2, 3], [3, 1, 4]
    assert reciprocal_rank_fusion([dense, sparse], n=4)[0] == 1
    assert set(reciprocal_rank_fusion([dense, sparse], n=4)) == {1, 2, 3, 4}


def test_rrf_with_one_list_keeps_its_order():
    assert reciprocal_rank_fusion([[5, 2, 9]], n=3) == [5, 2, 9]
