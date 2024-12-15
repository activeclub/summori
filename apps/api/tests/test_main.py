from api.main import load_rss


def test_load_rss():
    ret = load_rss("https://rss.arxiv.org/rss/cs.RO")
    assert ret["title"] == "cs.RO updates on arXiv.org"