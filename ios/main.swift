guard let model = try? YourModel(configuration: MLModelConfiguration()) else {
    fatalError("モデルの読み込み失敗")
}

guard let input = try? MLFeatureValue(cgImage: inputImage.cgImage!, constraint: model.model.inputDescriptionsByName["input_image"]!) else {
    fatalError("画像の変換失敗")
}

let inputDict = try? MLDictionaryFeatureProvider(dictionary: ["input_image": input])
let output = try? model.prediction(from: inputDict!)
let outputImage = output?.featureValue(for: "output_image")?.imageBufferValue
