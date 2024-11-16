document.addEventListener('DOMContentLoaded', function () {
    const btnAnnotation = document.getElementById('btn-annotation');
    const btnPrediction = document.getElementById('btn-prediction');
    const btnTableau = document.getElementById('btn-tableau');

    const predictionForm = document.getElementById('prediction-form');
    const imageContainer = document.getElementById('image-container');
    const tableContainer = document.getElementById('table-container');
    const downloadCsvBtn = document.getElementById('download-csv');

    btnAnnotation.addEventListener('click', function () {
        window.location.href = '/annotate';
    });

    btnPrediction.addEventListener('click', function () {
        predictionForm.style.display = 'block';
        imageContainer.style.display = 'none';
        tableContainer.style.display = 'none';
        downloadCsvBtn.style.display = 'none';
    });

    btnTableau.addEventListener('click', function () {
        tableContainer.style.display = 'block';
        downloadCsvBtn.style.display = 'block';
        predictionForm.style.display = 'none';
        imageContainer.style.display = 'none';
    });
});