module.exports = {
    resolve: {
      fallback: {
        fs: false,
        http: require.resolve("stream-http"),
        https: require.resolve("https-browserify"),
        url: require.resolve("url/")
      }
    }
  };
  