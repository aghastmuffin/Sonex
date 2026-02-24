from backbone.ltra import _mfa_aligner

_mfa_aligner.generate_aligned_v2("presiento", acoustic="spanish_mfa", dictionary="spanish_mfa", allow_fuzzy=True, fuzzy_max_lookahead=8)
