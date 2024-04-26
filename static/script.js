window.onload = function () {
  document.getElementById('file-sample').addEventListener('change', function (e) {
      var file = e.target.files[0];
      var blobUrl = window.URL.createObjectURL(file); // 選択された画像の一時的なURL
      var img = document.getElementById('file-preview');
      img.src = blobUrl;
  });
}
