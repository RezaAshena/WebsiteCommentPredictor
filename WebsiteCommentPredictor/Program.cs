using Microsoft.ML;
using WebsiteCommentPredictor;
using static Microsoft.ML.DataOperationsCatalog;

string _yelpDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
MLContext _mlContext = new MLContext();
TrainTestData splitDataView = LoadData(_mlContext);
ITransformer model = BuildAndTrainModel(_mlContext, splitDataView.TrainSet);

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

