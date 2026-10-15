# -*- coding: utf-8 -*-
# @Maintainer: zack zhao

# 用于测试cpu
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os
import sys
import traceback


def _record_smoke_exception():
    log_path = (os.environ.get("IVAS_BUILD_SMOKE_LOG") or os.environ.get("IVAT_BUILD_SMOKE_LOG"))
    if log_path:
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
            stream.write("\n")

try:
    from IVAS.main import main
except Exception:
    if any(
        flag in sys.argv
        for flag in (
            "--build-smoke-test",
            "--build-brand-smoke-test",
            "--build-video-workbench-smoke-test",
            "--build-sam31-smoke-test",
        )
    ):
        traceback.print_exc()
        _record_smoke_exception()
        raise SystemExit(1)
    raise

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        if any(
            flag in sys.argv
            for flag in (
                "--build-smoke-test",
                "--build-brand-smoke-test",
                "--build-video-workbench-smoke-test",
                "--build-sam31-smoke-test",
            )
        ):
            traceback.print_exc()
            _record_smoke_exception()
            raise SystemExit(1)
        raise
