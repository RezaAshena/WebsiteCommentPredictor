using Microsoft.ML;
using WebsiteCommentPredictor;
using static Microsoft.ML.DataOperationsCatalog;

string _yelpDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
MLContext _mlContext = new MLContext();
TrainTestData splitDataView = LoadData(_mlContext);
ITransformer model = BuildAndTrainModel(_mlContext, splitDataView.TrainSet);
GetPredictionForReviewContent(_mlContext, model, "This is an amazing product!");

var keepRunning = true;
while (keepRunning)
{
    Console.WriteLine("Enter a review to predict sentiment (or type 'exit' to quit):");
    var input = Console.ReadLine();
    if (input?.ToLower() == "exit")
    {
        keepRunning = false;
    }
    else if (!string.IsNullOrWhiteSpace(input))
    {
        GetPredictionForReviewContent(_mlContext, model, input);
    }
}
TrainTestData LoadData(MLContext mlContext)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_yelpDataPath);
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, 0.2);
    return splitDataView;

}

ITransformer BuildAndTrainModel(MLContext mlContext,IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));
    var model =estimator.Fit(splitTrainSet);
    return model;
}

void GetPredictionForReviewContent(MLContext mlContext, ITransformer model, string reviewContent)
{
    var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
    var sampleReview = new SentimentData { SentimentText = reviewContent };
    var prediction = predictionEngine.Predict(sampleReview);
    Console.WriteLine($"Review: {sampleReview.SentimentText}");
    Console.WriteLine($"Predicted sentiment: {(prediction.Prediction ? "Positive" : "Negative")}");
    Console.WriteLine($"Probability: {prediction.Probability}");
}