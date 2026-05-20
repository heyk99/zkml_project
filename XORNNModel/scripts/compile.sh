ezkl gen-settings \
--model artifacts/model.onnx \
--settings-path artifacts/setting.json

ezkl calibrate-settings \
--model artifacts/model.onnx \
--data data/input.json

ezkl compile-circuit \
--model artifacts/model.onnx \
--settings-path artifacts/setting.json \
--compiled-circuit artifacts/model.compiled
