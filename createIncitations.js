var maxId = db.data.count();

for (var i = 1; i <= maxId; i++) {
    var documentId = i.toString();
    db.inCitations.insert({id: documentId, citedBy: []});
}


for (i = 1; i <= maxId; i++) {
    documentId = i.toString();
    db.data.findOne({"id": documentId}).out_citations.forEach(citId => {
        print(citId, documentId);
        db.inCitations.update({id: citId},
            {$push: {citedBy: documentId}}
        )
    });
}
